# -*- coding: utf-8 -*-
"""
lesson11_2_double_dqn_overestimation.py

整体在干什么？
1) 构造一个专门用于展示“maximization bias（过估计偏差）”的极小 MDP：
   - 起点状态 A：LEFT 终止奖励0；RIGHT 转移到状态 B 奖励0
   - 状态 B：有 K 个动作，任何动作都会终止，但奖励是均值0的随机噪声（例如 N(0,1)）
   - 因此：真实 Q(A,LEFT)=0，Q(A,RIGHT)=0（两者无差异）
2) 用神经网络逼近 Q(s,a)（DQN风格），并对比两种 target：
   - Standard DQN: y = r + gamma * max_a Q_target(s',a)
   - Double DQN  : a* = argmax_a Q_online(s',a), y = r + gamma * Q_target(s',a*)
3) 训练过程中统计：
   - greedy 策略在状态 A 选择 RIGHT 的概率（带随机 tie-break）
   - Q(A,LEFT) 和 Q(A,RIGHT) 的估计值
   你应该看到：标准 DQN 更倾向 RIGHT（被“虚高的max”诱导），Double DQN 更接近 0.5（无偏/更弱偏差）。

代码组织要求（按你的课程要求）：
- 文件顶部注释说明整体目的
- 每个函数上方注释说明功能与逻辑
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Deque, List, Tuple
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 1) Environment: MaximizationBiasMDP
# -----------------------------
class MaximizationBiasMDP:
    """
    两步 MDP（专门用于展示 max 导致的过估计偏差）：

    状态：
      0 -> A（起点）
      1 -> B（噪声奖励状态）
      2 -> Terminal（终止）

    动作空间大小 = n_actions（例如 10）
    在状态 A：
      - action 0: LEFT  -> Terminal, reward=0
      - action 1: RIGHT -> B,        reward=0
      - action 2..n_actions-1: 视作 LEFT（Terminal, reward=0），简化实现
    在状态 B：
      - 任意动作 -> Terminal, reward ~ N(0, noise_std^2)（均值 0）
    """

    def __init__(self, n_actions: int = 10, noise_std: float = 1.0, seed: int = 0):
        self.n_actions = n_actions
        self.noise_std = noise_std
        self.rng = random.Random(seed)

        self.A = 0
        self.B = 1
        self.T = 2

    def reset(self) -> int:
        """重置到起点状态 A。"""
        return self.A

    def is_terminal(self, s: int) -> bool:
        """判断是否终止。"""
        return s == self.T

    def step(self, s: int, a: int) -> Tuple[int, float, bool]:
        """执行一步转移，返回 (s2, r, done)。"""
        if self.is_terminal(s):
            return s, 0.0, True

        if s == self.A:
            if a == 1:
                return self.B, 0.0, False
            # a==0 或 a>=2：都视作 LEFT
            return self.T, 0.0, True

        # s == B：任何动作都终止，但奖励是均值0噪声
        r = self.rng.gauss(0.0, self.noise_std)
        return self.T, r, True


# -----------------------------
# 2) Replay Buffer
# -----------------------------
@dataclass
class ReplayBuffer:
    """
    经验回放缓冲区：
    - push: 存储 transition
    - sample: 随机采样 batch（近似 i.i.d.）
    """

    capacity: int
    buffer: Deque[Tuple[int, int, float, int, bool]]

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        """存储一条经验。"""
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size: int, rng: random.Random):
        """随机采样一个 batch。"""
        batch = rng.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return list(s), list(a), list(r), list(s2), list(done)

    def __len__(self) -> int:
        return len(self.buffer)


# -----------------------------
# 3) Q Network
# -----------------------------
class QNetwork(nn.Module):
    """
    最小 MLP Q 网络：
      输入：state one-hot（dim = n_states）
      输出：Q(s,a)（dim = n_actions）
    """

    def __init__(self, n_states: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(n_states, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播输出所有动作的 Q 值。"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def to_onehot(states: List[int], n_states: int, device: torch.device) -> torch.Tensor:
    """离散状态列表 -> one-hot 张量。"""
    x = torch.zeros((len(states), n_states), device=device)
    for i, s in enumerate(states):
        x[i, s] = 1.0
    return x


def epsilon_greedy_action(
    q_net: QNetwork,
    s: int,
    epsilon: float,
    n_states: int,
    rng: random.Random,
    device: torch.device,
) -> int:
    """ε-greedy 选动作。"""
    n_actions = q_net.fc3.out_features
    if rng.random() < epsilon:
        return rng.randrange(n_actions)
    with torch.no_grad():
        q = q_net(to_onehot([s], n_states, device))[0]
        return int(torch.argmax(q).item())


def greedy_action_with_random_tie(q_values: torch.Tensor, rng: random.Random) -> int:
    """
    贪心动作（随机打破平局）：
    这样我们可以统计“选择 RIGHT 的概率”，而不是被固定 tie-break 影响。
    """
    max_v = float(q_values.max().item())
    idx = (q_values == max_v).nonzero(as_tuple=False).view(-1).tolist()
    return rng.choice(idx)


def compute_td_loss(
    online: QNetwork,
    target: QNetwork,
    batch,
    gamma: float,
    n_states: int,
    device: torch.device,
    use_double: bool,
) -> torch.Tensor:
    """
    计算 DQN / Double DQN 的 TD loss：

    Standard DQN:
      y = r + gamma * max_a Q_target(s',a)

    Double DQN:
      a* = argmax_a Q_online(s',a)
      y = r + gamma * Q_target(s',a*)

    共同点：
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
        if use_double:
            # Double DQN：online 选动作，target 评估
            a_star = online(x2).argmax(dim=1, keepdim=True)      # (batch,1)
            q_next = target(x2).gather(1, a_star).squeeze(1)     # (batch,)
        else:
            # Standard DQN：target 既选也评估（maximization bias 更强）
            q_next = target(x2).max(dim=1).values

        y = r_t + gamma * (1.0 - done_t) * q_next

    return F.mse_loss(q_sa, y)


def evaluate_p_choose_right_at_A(
    online: QNetwork,
    n_states: int,
    right_action: int,
    trials: int,
    seed: int,
    device: torch.device,
) -> float:
    """
    评估：在状态 A 上，用“贪心 + 随机平局打破”统计选择 RIGHT 的概率。
    """
    rng = random.Random(seed)
    A_state = 0
    with torch.no_grad():
        q = online(to_onehot([A_state], n_states, device))[0].cpu()
    cnt = 0
    for _ in range(trials):
        a = greedy_action_with_random_tie(q, rng)
        if a == right_action:
            cnt += 1
    return cnt / trials


def train_compare_dqn_vs_double(
    n_actions: int = 10,
    noise_std: float = 1.0,
    gamma: float = 1.0,
    n_steps: int = 60000,
    batch_size: int = 64,
    buffer_capacity: int = 20000,
    warmup: int = 500,
    lr: float = 1e-3,
    alpha_target_update_every: int = 500,
    epsilon: float = 0.1,
    seed: int = 0,
) -> None:
    """
    训练并对比：
      - Standard DQN（use_double=False）
      - Double DQN   （use_double=True）

    打印：
      - P(choose RIGHT at A)  （用 tie-random 的 greedy）
      - Q(A,LEFT) 与 Q(A,RIGHT) 的估计
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_dqn = MaximizationBiasMDP(n_actions=n_actions, noise_std=noise_std, seed=seed)
    env_ddqn = MaximizationBiasMDP(n_actions=n_actions, noise_std=noise_std, seed=seed + 1)

    n_states = 3  # A, B, T
    LEFT = 0
    RIGHT = 1

    def make_agent():
        online = QNetwork(n_states=n_states, n_actions=n_actions).to(device)
        target = QNetwork(n_states=n_states, n_actions=n_actions).to(device)
        target.load_state_dict(online.state_dict())
        optim = torch.optim.Adam(online.parameters(), lr=lr)
        buf = ReplayBuffer(capacity=buffer_capacity)
        return online, target, optim, buf

    online1, target1, optim1, buf1 = make_agent()
    online2, target2, optim2, buf2 = make_agent()

    checkpoints = [1000, 5000, 10000, 20000, 40000, n_steps]

    def step_train_one(
        env: MaximizationBiasMDP,
        online: QNetwork,
        target: QNetwork,
        optim: torch.optim.Optimizer,
        buf: ReplayBuffer,
        use_double: bool,
        t: int,
    ):
        """执行一次交互 + 可能的一次梯度更新 + 可能的 target 硬更新。"""
        s = env.reset()
        # 每个 episode 很短（A->T 或 A->B->T），我们直接按一次episode采样
        done = False
        while not done:
            a = epsilon_greedy_action(online, s, epsilon, n_states, rng, device)
            s2, r, done = env.step(s, a)
            buf.push(s, a, r, s2, done)
            s = s2

        # 更新一次
        if len(buf) >= max(warmup, batch_size):
            batch = buf.sample(batch_size, rng)
            loss = compute_td_loss(online, target, batch, gamma, n_states, device, use_double)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # target 硬更新
        if t % alpha_target_update_every == 0 and t > 0:
            target.load_state_dict(online.state_dict())

    # 主循环
    for t in range(1, n_steps + 1):
        step_train_one(env_dqn, online1, target1, optim1, buf1, use_double=False, t=t)
        step_train_one(env_ddqn, online2, target2, optim2, buf2, use_double=True, t=t)

        if t in checkpoints:
            with torch.no_grad():
                qA_dqn = online1(to_onehot([0], n_states, device))[0].cpu()
                qA_ddqn = online2(to_onehot([0], n_states, device))[0].cpu()

            p_right_dqn = evaluate_p_choose_right_at_A(online1, n_states, RIGHT, trials=2000, seed=123, device=device)
            p_right_ddqn = evaluate_p_choose_right_at_A(online2, n_states, RIGHT, trials=2000, seed=123, device=device)

            print(f"step={t:>6d}")
            print(f"  DQN    : P(RIGHT@A)={p_right_dqn:.3f} | Q(A,LEFT)={qA_dqn[LEFT]:+.3f} Q(A,RIGHT)={qA_dqn[RIGHT]:+.3f} | maxQ(A)={qA_dqn.max():+.3f}")
            print(f"  Double : P(RIGHT@A)={p_right_ddqn:.3f} | Q(A,LEFT)={qA_ddqn[LEFT]:+.3f} Q(A,RIGHT)={qA_ddqn[RIGHT]:+.3f} | maxQ(A)={qA_ddqn.max():+.3f}")

    # 额外：看看 B 状态的 maxQ（偏差的“来源点”）
    with torch.no_grad():
        qB_dqn = online1(to_onehot([1], n_states, device))[0].cpu()
        qB_ddqn = online2(to_onehot([1], n_states, device))[0].cpu()
    print("\nFinal check at state B (true value should be ~0):")
    print(f"  DQN    : max_a Q(B,a)={qB_dqn.max():+.3f} | mean_a Q(B,a)={qB_dqn.mean():+.3f}")
    print(f"  Double : max_a Q(B,a)={qB_ddqn.max():+.3f} | mean_a Q(B,a)={qB_ddqn.mean():+.3f}")


if __name__ == "__main__":
    print("Compare Standard DQN vs Double DQN on MaximizationBiasMDP")
    train_compare_dqn_vs_double(
        n_actions=10,
        noise_std=1.0,
        gamma=1.0,
        n_steps=60000,
        batch_size=64,
        buffer_capacity=20000,
        warmup=500,
        lr=1e-3,
        alpha_target_update_every=500,
        epsilon=0.1,
        seed=7,
    )
