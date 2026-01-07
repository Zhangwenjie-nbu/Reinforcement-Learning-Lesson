# -*- coding: utf-8 -*-
"""
lesson11_1_dqn_replay_and_target.py

整体在干什么？
1) 定义一个极小的离散环境 ChainMDP（N个状态，LEFT/RIGHT）：
   - 到达最右端：奖励 +1，done=True
   - 到达最左端：奖励 0，done=True
   - 其他步：奖励 -0.01
2) 实现最小 DQN：
   - Q网络：MLP，输入为 state 的 one-hot（维度 N）
   - 动作选择：epsilon-greedy
   - TD目标：y = r + gamma * max_a Q_target(s',a)
3) 两个稳定性机制（DQN关键）：
   - Replay Buffer：随机采样批量训练，降低样本相关性
   - Target Network：用慢更新网络计算目标，降低 moving target
4) 提供开关：use_replay / use_target，让你对比稳定性差异。

你需要掌握：
- Replay 解决“训练数据强相关/分布漂移”的问题（更像i.i.d.）
- Target Network 解决“自举目标随参数立即变化”的问题（moving target）
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
# 1) Environment: ChainMDP
# -----------------------------
class ChainMDP:
    """
    一个极小的离散MDP：
      - states: 0..n_states-1
      - start: middle
      - terminal: 0 (fail), n_states-1 (success)
      - reward: +1 at right terminal, 0 at left terminal, -0.01 per step otherwise
    """

    def __init__(self, n_states: int = 11, seed: int = 0):
        self.n_states = n_states
        self.start_state = n_states // 2
        self.rng = random.Random(seed)

    def reset(self) -> int:
        """重置到起点状态。"""
        return self.start_state

    def is_terminal(self, s: int) -> bool:
        """判断是否终止。"""
        return s == 0 or s == self.n_states - 1

    def step(self, s: int, a: int) -> Tuple[int, float, bool]:
        """
        执行动作：
          a=0: LEFT
          a=1: RIGHT
        """
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
    经验回放缓冲区：
    - push: 存储 transition
    - sample: 随机采样 batch（近似i.i.d.）
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
    最小MLP Q网络：
      输入：state one-hot (dim = n_states)
      输出：Q(s, a) for a in {0,1}
    """

    def __init__(self, n_states: int, hidden: int = 64, n_actions: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(n_states, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，输出Q值向量。"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def to_onehot(states: List[int], n_states: int, device: torch.device) -> torch.Tensor:
    """把离散状态列表转换为 one-hot 张量。"""
    x = torch.zeros((len(states), n_states), device=device)
    for i, s in enumerate(states):
        x[i, s] = 1.0
    return x


def epsilon_greedy_action(q_net: QNetwork, s: int, epsilon: float, n_states: int, rng: random.Random, device: torch.device) -> int:
    """ε-greedy 选动作。"""
    if rng.random() < epsilon:
        return rng.choice([0, 1])
    with torch.no_grad():
        x = to_onehot([s], n_states, device)
        q = q_net(x)[0]
        return int(torch.argmax(q).item())


def soft_update_target(target: QNetwork, online: QNetwork, tau: float) -> None:
    """
    软更新 target <- (1-tau)*target + tau*online
    这里提供给你对比；DQN经典是硬更新（每K步复制一次）。
    """
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)


def compute_td_loss(
    online: QNetwork,
    target: QNetwork,
    batch,
    gamma: float,
    n_states: int,
    device: torch.device,
    use_target: bool,
) -> torch.Tensor:
    """
    计算DQN的TD loss：
      y = r + gamma * max_a Q_target(s',a)   (done时y=r)
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
        if use_target:
            q_next = target(x2).max(dim=1).values
        else:
            # 不用target网络：目标直接依赖online网络（moving target更强）
            q_next = online(x2).max(dim=1).values

        y = r_t + gamma * (1.0 - done_t) * q_next

    return F.mse_loss(q_sa, y)


def train_dqn(
    use_replay: bool = True,
    use_target: bool = True,
    n_states: int = 11,
    n_episodes: int = 400,
    max_steps: int = 80,
    gamma: float = 0.99,
    lr: float = 1e-3,
    batch_size: int = 64,
    buffer_capacity: int = 5000,
    warmup: int = 200,
    target_update_every: int = 200,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 4000,
    seed: int = 0,
) -> None:
    """
    主训练过程（最小实现）：
    - 与环境交互收集transition
    - 若 use_replay：存入buffer并随机采样batch训练
      否则：直接用当前transition做一次更新（在线、强相关）
    - 若 use_target：用target网络计算TD目标，并定期硬更新
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = ChainMDP(n_states=n_states, seed=seed)
    online = QNetwork(n_states=n_states).to(device)
    target = QNetwork(n_states=n_states).to(device)
    target.load_state_dict(online.state_dict())

    optim = torch.optim.Adam(online.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_capacity)

    total_steps = 0
    returns_window = deque(maxlen=50)
    loss_window = deque(maxlen=200)

    def epsilon_by_step(t: int) -> float:
        """线性衰减epsilon。"""
        if t >= epsilon_decay_steps:
            return epsilon_end
        frac = t / epsilon_decay_steps
        return epsilon_start + frac * (epsilon_end - epsilon_start)

    for ep in range(1, n_episodes + 1):
        s = env.reset()
        ep_return = 0.0

        for _ in range(max_steps):
            eps = epsilon_by_step(total_steps)
            a = epsilon_greedy_action(online, s, eps, n_states, rng, device)
            s2, r, done = env.step(s, a)
            ep_return += r

            # 存入buffer（即使你不用replay，也存着方便切换）
            buffer.push(s, a, r, s2, done)

            # 训练一步
            if use_replay:
                if len(buffer) >= max(warmup, batch_size):
                    batch = buffer.sample(batch_size, rng)
                    loss = compute_td_loss(online, target, batch, gamma, n_states, device, use_target)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    loss_window.append(float(loss.item()))
            else:
                # 不用replay：用当前transition直接更新（样本强相关）
                batch = ([s], [a], [r], [s2], [done])
                loss = compute_td_loss(online, target, batch, gamma, n_states, device, use_target)
                optim.zero_grad()
                loss.backward()
                optim.step()
                loss_window.append(float(loss.item()))

            # target network 硬更新
            if use_target and (total_steps % target_update_every == 0) and total_steps > 0:
                target.load_state_dict(online.state_dict())

            s = s2
            total_steps += 1
            if done:
                break

        returns_window.append(ep_return)

        if ep in (1, 5, 10, 20, 50, 100, 200, n_episodes) or (ep % 50 == 0):
            avg_ret = sum(returns_window) / len(returns_window)
            avg_loss = (sum(loss_window) / len(loss_window)) if len(loss_window) > 0 else float("nan")
            print(
                f"ep={ep:>4} | avg_return(50)={avg_ret:>7.3f} | avg_loss(200)={avg_loss:>9.6f} "
                f"| replay={use_replay} target={use_target}"
            )


if __name__ == "__main__":
    print("=== Run 1: DQN with Replay + Target (recommended baseline) ===")
    train_dqn(use_replay=True, use_target=True, seed=7)

    print("\n=== Run 2: No Replay (online updates) + Target ===")
    train_dqn(use_replay=False, use_target=True, seed=7)

    print("\n=== Run 3: Replay + No Target (moving target) ===")
    train_dqn(use_replay=True, use_target=False, seed=7)
