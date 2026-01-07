# -*- coding: utf-8 -*-
"""
lesson11_6_n_step_return_dqn.py

整体在干什么？
1) 在简单链式环境 ChainMDP 上训练 DQN，并对比：
   - 1-step target（标准 DQN）
   - n-step target（multi-step return）
2) 实现关键改动点：把“单步 transition 存 buffer”改为“n-step transition 存 buffer”
   n-step transition 结构为：
     (s_t, a_t, R_t^{(n)}, s_{t+n}, done_{t+n})
   其中：
     R_t^{(n)} = sum_{k=0..n-1} gamma^k * r_{t+1+k}（遇到 done 提前截断）
3) TD target 改为：
     y = R^{(n)} + gamma^n * (1-done) * max_a Q_target(s_{t+n}, a)

你需要掌握：
- n-step 的收益：更快传播奖励（credit assignment 更快）
- n-step 的代价：n 越大方差越大，训练更“抖”
- 工程实现关键：用一个小队列缓存最近 n 步，凑齐时生成一条 n-step transition
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
    链式环境：
      states: 0..n_states-1
      start: middle
      terminal: 0 (reward=0), n_states-1 (reward=+1)
      step reward: -0.01 (鼓励尽快到达右端)
    actions:
      0 = LEFT, 1 = RIGHT
    """

    def __init__(self, n_states: int = 31, seed: int = 0):
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
# 2) Replay Buffer (store n-step transitions)
# -----------------------------
@dataclass
class ReplayBuffer:
    """
    经验回放缓冲区，存储 (s, a, Rn, s_next_n, done_n)。
    """

    capacity: int
    buffer: Deque[Tuple[int, int, float, int, bool]]

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, s: int, a: int, Rn: float, s2: int, done: bool) -> None:
        """存储一条 n-step transition。"""
        self.buffer.append((s, a, Rn, s2, done))

    def sample(self, batch_size: int, rng: random.Random):
        """随机采样 batch。"""
        batch = rng.sample(self.buffer, batch_size)
        s, a, Rn, s2, done = zip(*batch)
        return list(s), list(a), list(Rn), list(s2), list(done)

    def __len__(self) -> int:
        return len(self.buffer)


class NStepTransitionBuilder:
    """
    n-step transition 构造器（实现多步回报的关键组件）。

    逻辑：
    - 每走一步，把 (s, a, r, s2, done) 放进一个小队列 traj
    - 当 traj 长度达到 n 时，可以构造一条 n-step transition：
        s0 = traj[0].s
        a0 = traj[0].a
        Rn = r1 + gamma*r2 + ... + gamma^{n-1}*rn   （如中途done则截断）
        sN = traj[n-1].s2（或终止时的 s2）
        doneN = traj[n-1].done（或中途done）
    - 如果 episode 提前结束：要“把尾巴也吐出来”
      例如 n=3，但 episode 只剩 2 步：也应产生 2-step 的有效 transition（折扣到 done 截断）
      这在实践中是必要的，否则你会丢数据。
    """

    def __init__(self, n_step: int, gamma: float):
        self.n_step = n_step
        self.gamma = gamma
        self.traj: Deque[Tuple[int, int, float, int, bool]] = deque()

    def reset(self) -> None:
        """清空当前 episode 的缓存。"""
        self.traj.clear()

    def append_and_maybe_pop(self, s: int, a: int, r: float, s2: int, done: bool):
        """
        追加一步，并在可用时产生一条 n-step transition。
        返回：list of produced transitions（通常 0 或 1 条；在 done 时可能多条）
        """
        produced = []
        self.traj.append((s, a, r, s2, done))

        # 正常情况：凑够 n 步就产出一条
        if len(self.traj) >= self.n_step:
            produced.append(self._build_transition(n=self.n_step))
            # 滑动窗口：弹出最早一步
            self.traj.popleft()

        # 若 episode 结束：把尾巴全部吐出来（长度 < n 的也要）
        if done:
            while len(self.traj) > 0:
                produced.append(self._build_transition(n=len(self.traj)))
                self.traj.popleft()

        return produced

    def _build_transition(self, n: int) -> Tuple[int, int, float, int, bool]:
        """
        用 traj 前 n 个元素构造一条 n-step transition（若中途 done，会提前截断）。
        """
        s0, a0, _, _, _ = self.traj[0]

        Rn = 0.0
        sN = self.traj[n - 1][3]       # 默认第 n 步后的 next_state
        doneN = self.traj[n - 1][4]    # 默认第 n 步后的 done

        for k in range(n):
            _, _, r, s2, d = self.traj[k]
            Rn += (self.gamma ** k) * r
            if d:
                # 中途终止：截断
                sN = s2
                doneN = True
                break

        return s0, a0, float(Rn), int(sN), bool(doneN)


# -----------------------------
# 3) Q Network + utilities
# -----------------------------
class QNetwork(nn.Module):
    """最小MLP Q网络：输入 one-hot(state)，输出 Q(s,a) for a in {0,1}。"""

    def __init__(self, n_states: int, hidden: int = 128, n_actions: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(n_states, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播输出 Q 值向量。"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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


def epsilon_greedy_action(q_net: QNetwork, s: int, epsilon: float, n_states: int, rng: random.Random, device: torch.device) -> int:
    """ε-greedy 选动作。"""
    if rng.random() < epsilon:
        return rng.choice([0, 1])
    with torch.no_grad():
        q = q_net(to_onehot([s], n_states, device))[0]
        return int(torch.argmax(q).item())


# -----------------------------
# 4) n-step TD loss
# -----------------------------
def compute_n_step_td_loss(
    online: QNetwork,
    target: QNetwork,
    batch,
    gamma: float,
    n_step: int,
    n_states: int,
    device: torch.device,
) -> torch.Tensor:
    """
    n-step DQN TD loss：

      y = R^{(n)} + gamma^n * (1-done) * max_a Q_target(s_{t+n}, a)
      loss = MSE(Q_online(s_t, a_t), y)

    当 n_step=1 时退化为标准 1-step DQN。
    """
    s, a, Rn, s2, done = batch

    x = to_onehot(s, n_states, device)
    x2 = to_onehot(s2, n_states, device)

    a_t = torch.tensor(a, device=device, dtype=torch.long).unsqueeze(1)
    R_t = torch.tensor(Rn, device=device, dtype=torch.float32)
    done_t = torch.tensor(done, device=device, dtype=torch.float32)

    q_sa = online(x).gather(1, a_t).squeeze(1)

    with torch.no_grad():
        q_next = target(x2).max(dim=1).values
        y = R_t + (gamma ** n_step) * (1.0 - done_t) * q_next

    return F.mse_loss(q_sa, y)


# -----------------------------
# 5) Training loop
# -----------------------------
def train_dqn_n_step(
    n_step: int,
    n_states: int = 31,
    n_episodes: int = 900,
    max_steps: int = 140,
    gamma: float = 0.99,
    lr: float = 1e-3,
    batch_size: int = 64,
    buffer_capacity: int = 30000,
    warmup: int = 800,
    target_update_every: int = 500,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 14000,
    seed: int = 0,
) -> None:
    """
    训练 n-step DQN：
    - 与环境交互时，用 NStepTransitionBuilder 生成 n-step transition 存入 replay
    - 训练时使用 n-step target（gamma^n 的 bootstrap）
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

    n_builder = NStepTransitionBuilder(n_step=n_step, gamma=gamma)

    total_steps = 0
    returns_window = deque(maxlen=50)
    loss_window = deque(maxlen=200)
    p_right_window = deque(maxlen=200)

    checkpoints = set([1, 5, 10, 20, 50, 100, 200, 400, 700, n_episodes])

    for ep in range(1, n_episodes + 1):
        s = env.reset()
        n_builder.reset()
        ep_return = 0.0

        # 记录起点是否选 RIGHT（观察策略收敛）
        eps0 = epsilon_by_step(total_steps, eps_start, eps_end, eps_decay_steps)
        a0 = epsilon_greedy_action(online, s, eps0, n_states, rng, device)
        p_right_window.append(1 if a0 == 1 else 0)

        for _ in range(max_steps):
            eps = epsilon_by_step(total_steps, eps_start, eps_end, eps_decay_steps)
            a = epsilon_greedy_action(online, s, eps, n_states, rng, device)

            s2, r, done = env.step(s, a)
            ep_return += r

            # 关键：把单步 experience 交给 n-step builder，生成 0/1/多条 n-step transition
            produced = n_builder.append_and_maybe_pop(s, a, r, s2, done)
            for (s0, a0p, Rn, sNp, doneNp) in produced:
                buffer.push(s0, a0p, Rn, sNp, doneNp)

            # 训练一步
            if len(buffer) >= max(warmup, batch_size):
                batch = buffer.sample(batch_size, rng)
                loss = compute_n_step_td_loss(online, target, batch, gamma, n_step, n_states, device)
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
            avg_loss = (sum(loss_window) / len(loss_window)) if len(loss_window) else float("nan")
            p_right = sum(p_right_window) / len(p_right_window)
            print(f"[n={n_step}] ep={ep:>4} | avg_return(50)={avg_ret:>7.3f} | avg_loss(200)={avg_loss:>9.6f} | P(RIGHT@start)={p_right:>6.3f}")

    # 训练结束后打印起点 Q
    with torch.no_grad():
        q0 = online(to_onehot([env.start_state], n_states, device))[0].cpu()
    print(f"\n[n={n_step}] Q at start_state={env.start_state}: LEFT={q0[0]:+.3f}, RIGHT={q0[1]:+.3f}, maxQ={q0.max():+.3f}\n")


def main() -> None:
    """
    主入口：对比 1-step vs 3-step。
    """
    print("=== Run A: 1-step DQN (baseline) ===")
    train_dqn_n_step(n_step=1, seed=7)

    print("=== Run B: 3-step DQN (multi-step return) ===")
    train_dqn_n_step(n_step=3, seed=7)


if __name__ == "__main__":
    main()
