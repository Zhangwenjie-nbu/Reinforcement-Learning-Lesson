# -*- coding: utf-8 -*-
"""
lesson11_3_dueling_dqn.py

整体在干什么？
1) 定义一个“动作很多但多数无意义”的链式环境 MultiActionChainMDP：
   - states: 0..n_states-1
   - actions: 0=LEFT, 1=RIGHT, 2..K-1=NO-OP（不移动）
   - 到达最右端：reward=+1, done=True
   - 到达最左端：reward=0, done=True
   - 中间每步：reward=-0.01
2) 实现两种 Q 网络结构并对比：
   (A) StandardQNetwork：直接输出每个动作的 Q(s,a)
   (B) DuelingQNetwork ：先提取共享特征，再分别输出 V(s) 和 A(s,a)，并用
       Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a)) 聚合
3) 用同一套 DQN 训练逻辑（Replay Buffer + Target Network）分别训练两种网络，
   对比学习曲线（avg_return）。

你需要掌握：
- Dueling 不是改算法，是改网络表示，把 Q 拆成 V 与 A
- “减均值”是为了解决 V 与 A 的不可辨识性，稳定训练
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
# 1) Environment
# -----------------------------
class MultiActionChainMDP:
    """
    链式MDP，但动作数很多：
      - action 0: LEFT
      - action 1: RIGHT
      - action 2..K-1: NO-OP

    这样会制造“大量无意义动作”，让 dueling 架构更容易体现优势。
    """

    def __init__(self, n_states: int = 21, n_actions: int = 12, seed: int = 0):
        assert n_actions >= 3
        self.n_states = n_states
        self.n_actions = n_actions
        self.start_state = n_states // 2
        self.rng = random.Random(seed)

    def reset(self) -> int:
        """重置到起点状态。"""
        return self.start_state

    def is_terminal(self, s: int) -> bool:
        """终止条件：到达两端。"""
        return s == 0 or s == self.n_states - 1

    def step(self, s: int, a: int) -> Tuple[int, float, bool]:
        """执行一步转移，返回 (s2, r, done)。"""
        if self.is_terminal(s):
            return s, 0.0, True

        if a == 0:
            s2 = max(0, s - 1)
        elif a == 1:
            s2 = min(self.n_states - 1, s + 1)
        else:
            # NO-OP
            s2 = s

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


def epsilon_greedy_action(q_net: nn.Module, s: int, epsilon: float, n_states: int, n_actions: int, rng: random.Random, device: torch.device) -> int:
    """ε-greedy 选动作。"""
    if rng.random() < epsilon:
        return rng.randrange(n_actions)
    with torch.no_grad():
        q = q_net(to_onehot([s], n_states, device))[0]
        return int(torch.argmax(q).item())


# -----------------------------
# 4) Q Networks
# -----------------------------
class StandardQNetwork(nn.Module):
    """
    标准 DQN 网络：
      输入：state one-hot
      输出：Q(s,a) for all actions
    """

    def __init__(self, n_states: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(n_states, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播输出所有动作 Q 值。"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN 网络：
      - 共享特征提取层 feature(s)
      - Value head:      V(s)
      - Advantage head:  A(s,a)
      - 聚合：Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
    """

    def __init__(self, n_states: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.feat1 = nn.Linear(n_states, hidden)
        self.feat2 = nn.Linear(hidden, hidden)

        self.value1 = nn.Linear(hidden, hidden)
        self.value2 = nn.Linear(hidden, 1)

        self.adv1 = nn.Linear(hidden, hidden)
        self.adv2 = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播输出所有动作 Q 值（经过 dueling 聚合）。"""
        h = F.relu(self.feat1(x))
        h = F.relu(self.feat2(h))

        v = F.relu(self.value1(h))
        v = self.value2(v)  # (batch, 1)

        a = F.relu(self.adv1(h))
        a = self.adv2(a)    # (batch, n_actions)

        # 关键：减均值以解决不可辨识性
        a = a - a.mean(dim=1, keepdim=True)
        q = v + a
        return q


# -----------------------------
# 5) DQN Loss
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
    计算标准 DQN TD loss（用 target network 计算目标）：
      y = r + gamma * max_a Q_target(s',a)   (done时 y=r)
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
# 6) Training loop
# -----------------------------
def train_dqn_variant(
    net_type: str,
    n_states: int = 21,
    n_actions: int = 12,
    n_episodes: int = 700,
    max_steps: int = 100,
    gamma: float = 0.99,
    lr: float = 1e-3,
    batch_size: int = 64,
    buffer_capacity: int = 20000,
    warmup: int = 500,
    target_update_every: int = 300,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 8000,
    seed: int = 0,
) -> None:
    """
    训练一个 DQN 变体（Standard or Dueling）：
    - 使用 Replay Buffer 近似 i.i.d. 采样
    - 使用 Target Network 降低 moving target
    - 打印 avg_return 观察学习速度
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MultiActionChainMDP(n_states=n_states, n_actions=n_actions, seed=seed)

    if net_type == "standard":
        online = StandardQNetwork(n_states, n_actions).to(device)
        target = StandardQNetwork(n_states, n_actions).to(device)
    elif net_type == "dueling":
        online = DuelingQNetwork(n_states, n_actions).to(device)
        target = DuelingQNetwork(n_states, n_actions).to(device)
    else:
        raise ValueError("net_type must be 'standard' or 'dueling'")

    target.load_state_dict(online.state_dict())
    optim = torch.optim.Adam(online.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_capacity)

    total_steps = 0
    returns_window = deque(maxlen=50)
    loss_window = deque(maxlen=200)

    checkpoints = set([1, 5, 10, 20, 50, 100, 200, 400, n_episodes])
    for ep in range(1, n_episodes + 1):
        s = env.reset()
        ep_return = 0.0

        for _ in range(max_steps):
            eps = epsilon_by_step(total_steps, eps_start, eps_end, eps_decay_steps)
            a = epsilon_greedy_action(online, s, eps, n_states, n_actions, rng, device)
            s2, r, done = env.step(s, a)
            ep_return += r

            buffer.push(s, a, r, s2, done)

            # 训练一步（随机采样 batch）
            if len(buffer) >= max(warmup, batch_size):
                batch = buffer.sample(batch_size, rng)
                loss = compute_dqn_td_loss(online, target, batch, gamma, n_states, device)
                optim.zero_grad()
                loss.backward()
                optim.step()
                loss_window.append(float(loss.item()))

            # target 硬更新
            if total_steps > 0 and (total_steps % target_update_every == 0):
                target.load_state_dict(online.state_dict())

            s = s2
            total_steps += 1
            if done:
                break

        returns_window.append(ep_return)

        if (ep in checkpoints) or (ep % 50 == 0):
            avg_ret = sum(returns_window) / len(returns_window)
            avg_loss = (sum(loss_window) / len(loss_window)) if len(loss_window) > 0 else float("nan")
            print(f"[{net_type:>8}] ep={ep:>4} | avg_return(50)={avg_ret:>7.3f} | avg_loss(200)={avg_loss:>9.6f}")

    # 训练结束后，额外打印起点处的 Q 分布，看看 NO-OP 是否被压低
    with torch.no_grad():
        q0 = online(to_onehot([env.start_state], n_states, device))[0].cpu()
    print(f"\n[{net_type}] Q at start_state={env.start_state}:")
    print(f"  Q(LEFT)={q0[0]:+.3f}, Q(RIGHT)={q0[1]:+.3f}, Q(NO-OP mean)={q0[2:].mean():+.3f}, maxQ={q0.max():+.3f}\n")


def main() -> None:
    """
    主入口：在同一环境配置下对比 Standard DQN 与 Dueling DQN。
    """
    print("=== Standard DQN ===")
    train_dqn_variant(net_type="standard", seed=7)

    print("=== Dueling DQN ===")
    train_dqn_variant(net_type="dueling", seed=7)


if __name__ == "__main__":
    main()
