# -*- coding: utf-8 -*-
"""
lesson11_5_noisy_networks.py

整体在干什么？
1) 在一个简单链式环境 ChainMDP 上训练 DQN，并对比两种探索方式：
   (A) epsilon-greedy（动作层面随机）
   (B) NoisyNet（参数层面随机）：用 NoisyLinear 替代 Linear，并令 epsilon=0
2) 仍然使用 Replay Buffer + Target Network（稳定训练）
3) 训练过程中打印：
   - avg_return(50)：最近50个episode平均回报（学习速度/稳定性）
   - P(RIGHT@start)：在起点选择 RIGHT 的频率（策略收敛过程的一个可观测指标）

你需要掌握：
- NoisyNet 的探索来自“参数噪声”，而不是显式的 epsilon 随机动作
- NoisyLinear 里 sigma 是可学习的：网络能自动调探索强度
- 实践上通常：online 网络启用噪声以探索；target 网络可禁用噪声以稳定 TD target
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Deque, List, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 1) Environment: ChainMDP
# -----------------------------
class ChainMDP:
    """
    链式环境：
      states: 0..n_states-1
      start: middle
      terminal: 0 (reward=0), n_states-1 (reward=+1)
      step reward: -0.01 (鼓励尽快到达右端)
    actions:
      0 = LEFT, 1 = RIGHT
    """

    def __init__(self, n_states: int = 21, seed: int = 0):
        self.n_states = n_states
        self.start_state = n_states // 2
        self.rng = random.Random(seed)

    def reset(self) -> int:
        """重置到起点状态。"""
        return self.start_state

    def is_terminal(self, s: int) -> bool:
        """是否终止。"""
        return s == 0 or s == self.n_states - 1

    def step(self, s: int, a: int) -> Tuple[int, float, bool]:
        """执行动作，返回 (s2, r, done)。"""
        if self.is_terminal(s):
            return s, 0.0, True

        if a == 0:
            s2 = max(0, s - 1)
        else:
            s2 = min(self.n_states - 1, s + 1)

        done = self.is_terminal(s2)
        if s2 == self.n_states - 1:
            r = 1.0
        elif s2 == 0:
            r = 0.0
        else:
            r = -0.01
        return s2, r, done


# -----------------------------
# 2) Replay Buffer
# -----------------------------
@dataclass
class ReplayBuffer:
    """
    经验回放缓冲区：均匀随机采样。
    """

    capacity: int
    buffer: Deque[Tuple[int, int, float, int, bool]]

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        """存储一条 transition。"""
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size: int, rng: random.Random):
        """随机采样 batch。"""
        batch = rng.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return list(s), list(a), list(r), list(s2), list(done)

    def __len__(self) -> int:
        return len(self.buffer)


# -----------------------------
# 3) Utilities
# -----------------------------
def to_onehot(states: List[int], n_states: int, device: torch.device) -> torch.Tensor:
    """离散状态列表 -> one-hot 张量。"""
    x = torch.zeros((len(states), n_states), device=device)
    for i, s in enumerate(states):
        x[i, s] = 1.0
    return x


def epsilon_by_step(t: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    """线性衰减 epsilon。"""
    if t >= decay_steps:
        return eps_end
    frac = t / decay_steps
    return eps_start + frac * (eps_end - eps_start)


# -----------------------------
# 4) NoisyLinear (Factorized Gaussian)
# -----------------------------
class NoisyLinear(nn.Module):
    """
    NoisyLinear：W = W_mu + W_sigma ⊙ eps_W, b = b_mu + b_sigma ⊙ eps_b
    实现 factorized Gaussian noise（常用高效版本）。

    关键点：
    - weight_mu / bias_mu：可学习均值参数
    - weight_sigma / bias_sigma：可学习噪声幅度（探索强度）
    - reset_noise()：重采样噪声 eps
    - enable_noise/disable_noise：控制 forward 是否注入噪声
    """

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma0 = sigma0

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # 噪声buffer（不是参数）
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self._use_noise = True
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """
        初始化 mu 与 sigma。
        常见做法：
          mu ~ U[-1/sqrt(in), 1/sqrt(in)]
          sigma = sigma0 / sqrt(in)
        """
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        sigma_init = self.sigma0 / math.sqrt(self.in_features)
        self.weight_sigma.data.fill_(sigma_init)
        self.bias_sigma.data.fill_(sigma_init)

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        """
        factorized noise 的变换函数：
          f(x) = sign(x) * sqrt(|x|)
        """
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """重采样噪声 eps（factorized Gaussian）。"""
        eps_in = torch.randn(self.in_features, device=self.weight_mu.device)
        eps_out = torch.randn(self.out_features, device=self.weight_mu.device)
        eps_in = self._f(eps_in)
        eps_out = self._f(eps_out)

        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))
        self.bias_epsilon.copy_(eps_out)

    def enable_noise(self) -> None:
        """启用噪声注入（用于探索/训练）。"""
        self._use_noise = True

    def disable_noise(self) -> None:
        """禁用噪声注入（用于更稳定的 target 或评估）。"""
        self._use_noise = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：根据 _use_noise 决定是否注入噪声。"""
        if self._use_noise and self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


# -----------------------------
# 5) Q Networks
# -----------------------------
class StandardQNetwork(nn.Module):
    """标准 DQN Q 网络：MLP（Linear + ReLU）。"""

    def __init__(self, n_states: int, hidden: int = 128, n_actions: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(n_states, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输出 Q(s,·)。"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class NoisyQNetwork(nn.Module):
    """
    NoisyNet Q 网络：把 Linear 替换成 NoisyLinear。
    注意：
    - 需要提供 reset_noise() 以便在行动/训练时重采样噪声
    - 需要 enable_noise/disable_noise 方便控制 target 的稳定性
    """

    def __init__(self, n_states: int, hidden: int = 128, n_actions: int = 2, sigma0: float = 0.5):
        super().__init__()
        self.fc1 = NoisyLinear(n_states, hidden, sigma0=sigma0)
        self.fc2 = NoisyLinear(hidden, hidden, sigma0=sigma0)
        self.fc3 = NoisyLinear(hidden, n_actions, sigma0=sigma0)

    def reset_noise(self) -> None:
        """重采样所有 NoisyLinear 的噪声。"""
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()

    def enable_noise(self) -> None:
        """启用所有层噪声。"""
        self.fc1.enable_noise()
        self.fc2.enable_noise()
        self.fc3.enable_noise()

    def disable_noise(self) -> None:
        """禁用所有层噪声。"""
        self.fc1.disable_noise()
        self.fc2.disable_noise()
        self.fc3.disable_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输出 Q(s,·)。"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# -----------------------------
# 6) Action selection
# -----------------------------
def epsilon_greedy_action(
    q_net: nn.Module,
    s: int,
    epsilon: float,
    n_states: int,
    rng: random.Random,
    device: torch.device,
) -> int:
    """ε-greedy 选动作（用于标准 DQN）。"""
    if rng.random() < epsilon:
        return rng.choice([0, 1])
    with torch.no_grad():
        q = q_net(to_onehot([s], n_states, device))[0]
        return int(torch.argmax(q).item())


def greedy_action_noisy(
    q_net: NoisyQNetwork,
    s: int,
    n_states: int,
    device: torch.device,
    resample_noise: bool = True,
) -> int:
    """
    NoisyNet 贪心动作：
    - 不用 epsilon，随机性来自网络参数噪声
    - 常见做法：每次选动作前重采样一次噪声，使策略持续探索
    """
    if resample_noise:
        q_net.reset_noise()
    with torch.no_grad():
        q = q_net(to_onehot([s], n_states, device))[0]
        return int(torch.argmax(q).item())


# -----------------------------
# 7) TD loss
# -----------------------------
def compute_dqn_td_loss(
    online: nn.Module,
    target: nn.Module,
    batch,
    gamma: float,
    n_states: int,
    device: torch.device,
) -> torch.Tensor:
    """
    DQN TD loss（使用 target network）：
      y = r + gamma * (1-done) * max_a Q_target(s',a)
      loss = MSE(Q_online(s,a), y)
    """
    s, a, r, s2, done = batch

    x = to_onehot(s, n_states, device)
    x2 = to_onehot(s2, n_states, device)

    a_t = torch.tensor(a, device=device, dtype=torch.long).unsqueeze(1)
    r_t = torch.tensor(r, device=device, dtype=torch.float32)
    done_t = torch.tensor(done, device=device, dtype=torch.float32)

    q_sa = online(x).gather(1, a_t).squeeze(1)

    with torch.no_grad():
        q_next = target(x2).max(dim=1).values
        y = r_t + gamma * (1.0 - done_t) * q_next

    return F.mse_loss(q_sa, y)


# -----------------------------
# 8) Training loop
# -----------------------------
def train_compare_eps_vs_noisy(
    n_states: int = 21,
    n_episodes: int = 700,
    max_steps: int = 120,
    gamma: float = 0.99,
    lr: float = 1e-3,
    batch_size: int = 64,
    buffer_capacity: int = 20000,
    warmup: int = 500,
    target_update_every: int = 400,
    # epsilon-greedy schedule (for standard DQN)
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 10000,
    # noisy params
    sigma0: float = 0.5,
    seed: int = 0,
) -> None:
    """
    对比两种探索策略：
    A) Standard DQN + epsilon-greedy
    B) NoisyNet DQN + epsilon=0（仅靠参数噪声探索）

    设计要点（为什么这么做）：
    - 两者都用 Replay + Target：把差异尽可能集中在“探索机制”
    - NoisyNet：online 启用噪声用于探索与训练；target 禁用噪声用于稳定 TD target
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_one(tag: str, use_noisy: bool):
        env = ChainMDP(n_states=n_states, seed=seed)
        buffer = ReplayBuffer(capacity=buffer_capacity)

        if use_noisy:
            online = NoisyQNetwork(n_states=n_states, hidden=128, n_actions=2, sigma0=sigma0).to(device)
            target = NoisyQNetwork(n_states=n_states, hidden=128, n_actions=2, sigma0=sigma0).to(device)
            online.enable_noise()
            target.disable_noise()  # 关键：target 不注入噪声，目标更稳定
        else:
            online = StandardQNetwork(n_states=n_states, hidden=128, n_actions=2).to(device)
            target = StandardQNetwork(n_states=n_states, hidden=128, n_actions=2).to(device)

        target.load_state_dict(online.state_dict())
        optim = torch.optim.Adam(online.parameters(), lr=lr)

        total_steps = 0
        returns_window = deque(maxlen=50)
        loss_window = deque(maxlen=200)

        start_right_window = deque(maxlen=200)  # 统计起点是否选RIGHT

        checkpoints = set([1, 5, 10, 20, 50, 100, 200, 400, n_episodes])

        for ep in range(1, n_episodes + 1):
            s = env.reset()
            ep_return = 0.0

            # 记录起点动作（观察探索/收敛）
            if use_noisy:
                a0 = greedy_action_noisy(online, s, n_states, device, resample_noise=True)
            else:
                eps = epsilon_by_step(total_steps, eps_start, eps_end, eps_decay_steps)
                a0 = epsilon_greedy_action(online, s, eps, n_states, rng, device)
            start_right_window.append(1 if a0 == 1 else 0)

            for step in range(max_steps):
                if use_noisy:
                    # NoisyNet：不使用 epsilon；每步重采样一次噪声得到“不同策略”的一致探索
                    a = greedy_action_noisy(online, s, n_states, device, resample_noise=True)
                else:
                    eps = epsilon_by_step(total_steps, eps_start, eps_end, eps_decay_steps)
                    a = epsilon_greedy_action(online, s, eps, n_states, rng, device)

                s2, r, done = env.step(s, a)
                ep_return += r
                buffer.push(s, a, r, s2, done)

                # 训练一步
                if len(buffer) >= max(warmup, batch_size):
                    batch = buffer.sample(batch_size, rng)

                    # NoisyNet：训练时也重采样噪声，让梯度对应“当前噪声实例”的目标
                    if use_noisy:
                        online.reset_noise()

                    loss = compute_dqn_td_loss(online, target, batch, gamma, n_states, device)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    loss_window.append(float(loss.item()))

                # target 硬更新
                if total_steps > 0 and (total_steps % target_update_every == 0):
                    target.load_state_dict(online.state_dict())
                    if use_noisy:
                        target.disable_noise()

                s = s2
                total_steps += 1
                if done:
                    break

            returns_window.append(ep_return)

            if (ep in checkpoints) or (ep % 50 == 0):
                avg_ret = sum(returns_window) / len(returns_window)
                avg_loss = (sum(loss_window) / len(loss_window)) if len(loss_window) else float("nan")
                p_right = sum(start_right_window) / len(start_right_window)
                print(f"[{tag:>8}] ep={ep:>4} | avg_return(50)={avg_ret:>7.3f} | avg_loss(200)={avg_loss:>9.6f} | P(RIGHT@start)={p_right:>6.3f}")

        # 训练结束后，看看起点 Q 值
        with torch.no_grad():
            if use_noisy:
                online.disable_noise()  # 评估时关噪声，读 mu 的“平均策略”
            q0 = online(to_onehot([env.start_state], n_states, device))[0].cpu()
        print(f"\n[{tag}] Q at start_state={env.start_state}: LEFT={q0[0]:+.3f}, RIGHT={q0[1]:+.3f}, maxQ={q0.max():+.3f}\n")

    print("=== Run A: Standard DQN with epsilon-greedy ===")
    run_one(tag="EPS", use_noisy=False)

    print("=== Run B: NoisyNet DQN (epsilon=0, exploration via parameter noise) ===")
    run_one(tag="NOISY", use_noisy=True)


def main() -> None:
    """主入口：运行对比实验。"""
    train_compare_eps_vs_noisy(seed=7)


if __name__ == "__main__":
    main()
