# -*- coding: utf-8 -*-
"""
lesson11_4_prioritized_replay_per.py

整体在干什么？
1) 定义一个简单链式环境 ChainMDP（与前面类似）：
   - 到达右端：+1 终止
   - 到达左端：0 终止
   - 中间每步：-0.01
2) 实现两种 replay buffer：
   (A) UniformReplayBuffer：均匀随机采样
   (B) PrioritizedReplayBuffer（PER）：
       - priority p_i = |TD-error| + eps
       - sampling prob P(i) ∝ p_i^alpha
       - IS weight w_i = (1/N * 1/P(i))^beta，并归一化到 [0,1]
       - 每次训练后用新 TD-error 回写更新 priorities
3) 在同一套 DQN（含 target network）训练逻辑下，对比：
   - avg_return 曲线
   - loss/TD-error 的变化趋势（可辅助观察PER效果）

你需要掌握：
- PER 的“快”来自：更多抽到高TD-error样本
- PER 的“偏差”来自：采样分布从均匀变成 P(i)
- IS correction 的“修正”来自：用 w_i 对 loss 加权，beta 逐步退火到 1
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Deque, List, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 1) Environment
# -----------------------------
class ChainMDP:
    """
    一个简单链式MDP：
      - states: 0..n_states-1
      - start: middle
      - terminal: 0 (fail), n_states-1 (success)
      - reward: +1 at right terminal, 0 at left terminal, -0.01 per step otherwise
    """

    def __init__(self, n_states: int = 21, seed: int = 0):
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
# 2) Replay Buffers
# -----------------------------
@dataclass
class UniformReplayBuffer:
    """
    均匀采样 replay buffer：
    - push: 存储 transition
    - sample: 均匀随机抽 batch
    """
    capacity: int
    buffer: Deque[Tuple[int, int, float, int, bool]]

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        """存储 transition。"""
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size: int, rng: random.Random):
        """均匀采样，返回 batch + indices + weights（weights全1）。"""
        batch = rng.sample(self.buffer, batch_size)
        idxs = None
        weights = np.ones(batch_size, dtype=np.float32)
        s, a, r, s2, done = zip(*batch)
        return list(s), list(a), list(r), list(s2), list(done), idxs, weights

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    PER（朴素版，O(N) 概率采样，适合教学与小规模实验）：

    核心思想：
    - 每条样本 i 有 priority p_i（初始可设为当前最大值，保证新样本会被抽到）
    - 采样概率 P(i) ∝ p_i^alpha
    - IS 权重 w_i = (1/N * 1/P(i))^beta，并归一化（除以 max w）
    - 训练后用新 TD-error 更新 priority

    注意：
    - alpha 控制“偏向程度”
    - beta 控制“偏差校正强度”，通常随训练步数从 beta0 退火到 1
    """

    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-3, seed: int = 0):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.rng = np.random.default_rng(seed)

        self.pos = 0
        self.size = 0

        self.s = np.zeros(capacity, dtype=np.int64)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros(capacity, dtype=np.int64)
        self.done = np.zeros(capacity, dtype=np.bool_)

        # priority 存储
        self.p = np.zeros(capacity, dtype=np.float32)
        self.max_p = 1.0

    def push(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        """写入一条经验；新样本 priority 设为当前 max_p（保证新数据能被抽到）。"""
        self.s[self.pos] = s
        self.a[self.pos] = a
        self.r[self.pos] = r
        self.s2[self.pos] = s2
        self.done[self.pos] = done

        self.p[self.pos] = self.max_p

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float):
        """
        按 P(i) ∝ p_i^alpha 采样 batch，并返回 IS weights。
        返回：
          s, a, r, s2, done, idxs, weights
        """
        assert self.size >= batch_size

        p = self.p[: self.size] + self.eps
        probs = p ** self.alpha
        probs = probs / probs.sum()

        idxs = self.rng.choice(self.size, size=batch_size, replace=False, p=probs)

        # IS weights
        # w_i = (1/N * 1/P(i))^beta
        N = self.size
        w = (1.0 / (N * probs[idxs])) ** beta
        w = w / w.max()  # 归一化到 (0,1]
        w = w.astype(np.float32)

        s = self.s[idxs].tolist()
        a = self.a[idxs].tolist()
        r = self.r[idxs].tolist()
        s2 = self.s2[idxs].tolist()
        done = self.done[idxs].tolist()
        return s, a, r, s2, done, idxs, w

    def update_priorities(self, idxs: np.ndarray, new_priorities: np.ndarray) -> None:
        """用新的 TD-error 更新 priority，并维护 max_p。"""
        new_p = np.abs(new_priorities).astype(np.float32) + self.eps
        self.p[idxs] = new_p
        self.max_p = max(self.max_p, float(new_p.max()))

    def __len__(self) -> int:
        return self.size


# -----------------------------
# 3) Q Network + helpers
# -----------------------------
class QNetwork(nn.Module):
    """最小MLP Q网络：输入 one-hot(state)，输出 Q(s, a) for a in {0,1}。"""

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


def beta_by_step(t: int, beta_start: float, beta_end: float, anneal_steps: int) -> float:
    """线性退火 beta 到 1（或 beta_end）。"""
    if t >= anneal_steps:
        return beta_end
    frac = t / anneal_steps
    return beta_start + frac * (beta_end - beta_start)


def epsilon_greedy_action(q_net: QNetwork, s: int, epsilon: float, n_states: int, rng: random.Random, device: torch.device) -> int:
    """ε-greedy 选动作。"""
    if rng.random() < epsilon:
        return rng.choice([0, 1])
    with torch.no_grad():
        q = q_net(to_onehot([s], n_states, device))[0]
        return int(torch.argmax(q).item())


def compute_td_loss_and_td_error(
    online: QNetwork,
    target: QNetwork,
    batch,
    gamma: float,
    n_states: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算每个样本的 TD-error 与加权 loss：

    输入 batch：
      s, a, r, s2, done, idxs, weights

    目标：
      y = r + gamma * (1-done) * max_a Q_target(s2,a)
    TD-error：
      delta = y - Q_online(s,a)

    loss：
      mean( weights * delta^2 )

    返回：
      loss, td_error(detached)
    """
    s, a, r, s2, done, idxs, weights = batch

    x = to_onehot(s, n_states, device)
    x2 = to_onehot(s2, n_states, device)

    a_t = torch.tensor(a, device=device, dtype=torch.long).unsqueeze(1)
    r_t = torch.tensor(r, device=device, dtype=torch.float32)
    done_t = torch.tensor(done, device=device, dtype=torch.float32)

    w_t = torch.tensor(weights, device=device, dtype=torch.float32)

    q_sa = online(x).gather(1, a_t).squeeze(1)

    with torch.no_grad():
        q_next = target(x2).max(dim=1).values
        y = r_t + gamma * (1.0 - done_t) * q_next

    td_error = y - q_sa
    loss = (w_t * (td_error ** 2)).mean()
    return loss, td_error.detach()


# -----------------------------
# 4) Training
# -----------------------------
def train_dqn(
    use_per: bool,
    n_states: int = 21,
    n_episodes: int = 900,
    max_steps: int = 120,
    gamma: float = 0.99,
    lr: float = 1e-3,
    batch_size: int = 64,
    buffer_capacity: int = 20000,
    warmup: int = 500,
    target_update_every: int = 400,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 12000,
    # PER 超参
    per_alpha: float = 0.6,
    per_beta_start: float = 0.4,
    per_beta_end: float = 1.0,
    per_beta_anneal_steps: int = 20000,
    seed: int = 0,
) -> None:
    """
    训练 DQN：
    - 始终使用 target network（稳定）
    - replay 选择：uniform 或 PER
    - PER 时：采样返回 IS weights；每次更新后把 TD-error 写回 priorities
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = ChainMDP(n_states=n_states, seed=seed)
    online = QNetwork(n_states=n_states).to(device)
    target = QNetwork(n_states=n_states).to(device)
    target.load_state_dict(online.state_dict())

    optim = torch.optim.Adam(online.parameters(), lr=lr)

    if use_per:
        buffer = PrioritizedReplayBuffer(capacity=buffer_capacity, alpha=per_alpha, eps=1e-3, seed=seed)
    else:
        buffer = UniformReplayBuffer(capacity=buffer_capacity)

    total_steps = 0
    returns_window = deque(maxlen=50)
    loss_window = deque(maxlen=200)
    abs_td_window = deque(maxlen=200)

    checkpoints = set([1, 5, 10, 20, 50, 100, 200, 400, 700, n_episodes])

    for ep in range(1, n_episodes + 1):
        s = env.reset()
        ep_return = 0.0

        for _ in range(max_steps):
            eps = epsilon_by_step(total_steps, eps_start, eps_end, eps_decay_steps)
            a = epsilon_greedy_action(online, s, eps, n_states, rng, device)
            s2, r, done = env.step(s, a)
            ep_return += r

            buffer.push(s, a, r, s2, done)

            # 训练一步：从 buffer 采样
            if len(buffer) >= max(warmup, batch_size):
                if use_per:
                    beta = beta_by_step(total_steps, per_beta_start, per_beta_end, per_beta_anneal_steps)
                    batch = buffer.sample(batch_size=batch_size, beta=beta)
                else:
                    batch = buffer.sample(batch_size=batch_size, rng=rng)

                loss, td_error = compute_td_loss_and_td_error(online, target, batch, gamma, n_states, device)

                optim.zero_grad()
                loss.backward()
                optim.step()

                loss_window.append(float(loss.item()))
                abs_td_window.append(float(td_error.abs().mean().item()))

                # PER：回写 priorities（用 |TD-error|）
                if use_per:
                    _, _, _, _, _, idxs, _ = batch
                    buffer.update_priorities(idxs, td_error.cpu().numpy())

            # target network 硬更新
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
            avg_abs_td = (sum(abs_td_window) / len(abs_td_window)) if len(abs_td_window) else float("nan")
            tag = "PER" if use_per else "Uniform"
            print(f"[{tag:>7}] ep={ep:>4} | avg_return(50)={avg_ret:>7.3f} | avg_loss(200)={avg_loss:>9.6f} | avg|TD|(200)={avg_abs_td:>8.5f}")

    # 打印起点处Q
    with torch.no_grad():
        q0 = online(to_onehot([env.start_state], n_states, device))[0].cpu()
    tag = "PER" if use_per else "Uniform"
    print(f"\n[{tag}] Q at start_state={env.start_state}: LEFT={q0[0]:+.3f}, RIGHT={q0[1]:+.3f}, maxQ={q0.max():+.3f}\n")


def main() -> None:
    """
    主入口：对比 Uniform replay 与 PER。
    """
    print("=== Run 1: Uniform Replay ===")
    train_dqn(use_per=False, seed=7)

    print("=== Run 2: Prioritized Experience Replay (PER) + IS correction ===")
    train_dqn(use_per=True, seed=7)


if __name__ == "__main__":
    main()
