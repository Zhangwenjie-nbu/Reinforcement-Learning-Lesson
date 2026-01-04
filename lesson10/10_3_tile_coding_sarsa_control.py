# -*- coding: utf-8 -*-
"""
lesson10_3_tile_coding_sarsa_control.py

整体在干什么？
1) 环境：连续状态 ContinuousLineWorld
   - 状态 x ∈ [0,1]
   - 动作 LEFT/RIGHT
   - 到达 x>=1 终止并奖励 +1；到达 x<=0 终止奖励 0
2) 特征：Tile Coding（多组错位 tilings），对每个 x 产生 M 个激活 tile id（稀疏）。
3) 构造 state-action 特征 phi(x,a)：
   - 每个动作一套独立参数（权重向量分块）
   - 对于动作 a，激活特征索引为 a*F + tile_id
4) 算法：Semi-gradient SARSA（on-policy control）
   - 行为策略：ε-greedy 基于当前 Q(x,a)
   - 更新：
       delta = r + gamma*Q(x',a') - Q(x,a)
       对 (x,a) 的激活权重：w[i] += (alpha/M) * delta
5) 训练过程中周期评估：
   - 用纯贪心策略（ε=0）跑多次，统计到达右端的成功率

你需要掌握：
- 从 V 到 Q 的关键变化：需要比较动作，因此学习 Q(x,a)
- Tile coding 下 Q 是稀疏线性和，更新只影响少量权重（局部泛化）
- SARSA 是 on-policy：目标里的 a' 来自同一条 ε-greedy 行为策略
"""

import random
import math
from dataclasses import dataclass
from typing import List, Tuple

LEFT, RIGHT = 0, 1
ACTIONS = [LEFT, RIGHT]


class ContinuousLineWorld:
    """
    连续 1D 环境：
    - x ∈ [0,1]
    - 动作：LEFT/RIGHT
    - 动力学：x <- x ± step + noise
    - 终止：x<=0 或 x>=1
    - 奖励：到达右端 (x>=1) 给 +1，否则 0
    """

    def __init__(self, start_x=0.5, step=0.05, noise_std=0.01, seed=0):
        self.start_x = start_x
        self.step = step
        self.noise_std = noise_std
        self.rng = random.Random(seed)

    def reset(self) -> float:
        """重置到起点位置。"""
        return self.start_x

    def is_terminal(self, x: float) -> bool:
        """是否到达终止边界。"""
        return x <= 0.0 or x >= 1.0

    def step_env(self, x: float, a: int) -> Tuple[float, float, bool]:
        """执行动作，返回 (x2, r, done)。"""
        if self.is_terminal(x):
            return x, 0.0, True

        direction = -1.0 if a == LEFT else 1.0
        noise = self.rng.gauss(0.0, self.noise_std)
        x2 = x + direction * self.step + noise

        # clamp
        if x2 < 0.0:
            x2 = 0.0
        if x2 > 1.0:
            x2 = 1.0

        done = self.is_terminal(x2)
        r = 1.0 if x2 >= 1.0 else 0.0
        return x2, r, done


@dataclass
class TileCoder1D:
    """
    1D Tile Coder（显式索引版，便于理解）
    - num_tilings: M
    - tiles_per_tiling: N
    总状态特征数 F = M*N
    """

    num_tilings: int
    tiles_per_tiling: int
    x_min: float = 0.0
    x_max: float = 1.0

    def __post_init__(self):
        self.width = (self.x_max - self.x_min) / self.tiles_per_tiling

    def active_features(self, x: float) -> List[int]:
        """
        给定 x，返回状态激活特征（长度= M）。
        """
        x = min(max(x, self.x_min), self.x_max - 1e-12)

        feats = []
        for t in range(self.num_tilings):
            offset = (t / self.num_tilings) * self.width
            scaled = (x - self.x_min + offset) / self.width
            tile_index = int(math.floor(scaled))
            tile_index = max(0, min(self.tiles_per_tiling - 1, tile_index))
            feats.append(t * self.tiles_per_tiling + tile_index)
        return feats

    @property
    def n_state_features(self) -> int:
        """状态特征总数 F。"""
        return self.num_tilings * self.tiles_per_tiling


@dataclass
class TileQFunction:
    """
    基于 tile coding 的动作价值函数：
    - 每个动作一套权重：总参数数 = |A| * F
    - Q(x,a) = sum_{i in active(x,a)} w[i]
    """

    tile_coder: TileCoder1D
    w: List[float]

    def q_value(self, x: float, a: int) -> float:
        """计算 Q(x,a)。"""
        feats = self._active_sa_features(x, a)
        return sum(self.w[i] for i in feats)

    def update(self, x: float, a: int, delta: float, alpha: float):
        """对 (x,a) 的激活权重做 SARSA 更新。"""
        feats = self._active_sa_features(x, a)
        step = alpha / self.tile_coder.num_tilings  # 常用缩放
        for i in feats:
            self.w[i] += step * delta

    def _active_sa_features(self, x: float, a: int) -> List[int]:
        """
        把 state features 映射到 state-action features：
        feat_id_sa = a*F + feat_id_state
        """
        F = self.tile_coder.n_state_features
        base = a * F
        return [base + fid for fid in self.tile_coder.active_features(x)]

    @property
    def n_params(self) -> int:
        """总参数数 |A|*F。"""
        return len(ACTIONS) * self.tile_coder.n_state_features


def epsilon_greedy_action(qf: TileQFunction, x: float, epsilon: float, rng: random.Random) -> int:
    """
    ε-greedy 行为策略：
    - ε：探索
    - 1-ε：贪心
    """
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)
    qL = qf.q_value(x, LEFT)
    qR = qf.q_value(x, RIGHT)
    return RIGHT if qR >= qL else LEFT


def greedy_action(qf: TileQFunction, x: float) -> int:
    """纯贪心动作（用于评估）。"""
    qL = qf.q_value(x, LEFT)
    qR = qf.q_value(x, RIGHT)
    return RIGHT if qR >= qL else LEFT


def evaluate_greedy_success(env_seed: int, qf: TileQFunction, n_trials=300, max_steps=300) -> float:
    """
    评估：用贪心策略跑 n_trials 次，返回到达右端 (x>=1) 的成功率。
    使用独立 seed 的环境，避免训练随机性污染评估。
    """
    env = ContinuousLineWorld(start_x=0.5, step=0.05, noise_std=0.01, seed=env_seed)
    success = 0
    for _ in range(n_trials):
        x = env.reset()
        for _ in range(max_steps):
            if env.is_terminal(x):
                break
            a = greedy_action(qf, x)
            x, r, done = env.step_env(x, a)
            if done:
                if x >= 1.0:
                    success += 1
                break
    return success / n_trials


def train_sarsa_tile_coding(
    env: ContinuousLineWorld,
    tile_coder: TileCoder1D,
    gamma=0.99,
    alpha=0.2,
    epsilon=0.1,
    n_episodes=20000,
    max_steps=300,
    seed=0,
):
    """
    Semi-gradient SARSA 主训练循环（on-policy control）：
    - 初始化权重 w=0
    - 每条 episode：
        x0=reset
        a0=ε-greedy(Q,x0)
        repeat:
            x', r, done = step(x,a)
            a' = ε-greedy(Q,x')  (若 done 则不需要)
            delta = r + gamma*Q(x',a') - Q(x,a)   (终止时 gamma*Q=0)
            update(x,a,delta)
            x,a <- x',a'
    """
    rng = random.Random(seed)
    qf = TileQFunction(tile_coder=tile_coder, w=[0.0] * (len(ACTIONS) * tile_coder.n_state_features))

    checkpoints = [200, 1000, 5000, 10000, n_episodes]

    for ep in range(1, n_episodes + 1):
        x = env.reset()
        a = epsilon_greedy_action(qf, x, epsilon, rng)

        for _ in range(max_steps):
            x2, r, done = env.step_env(x, a)

            q_xa = qf.q_value(x, a)

            if done:
                td_target = r
                delta = td_target - q_xa
                qf.update(x, a, delta, alpha)
                break

            a2 = epsilon_greedy_action(qf, x2, epsilon, rng)
            td_target = r + gamma * qf.q_value(x2, a2)
            delta = td_target - q_xa
            qf.update(x, a, delta, alpha)

            x, a = x2, a2

        if ep in checkpoints:
            sr = evaluate_greedy_success(env_seed=999, qf=qf, n_trials=400, max_steps=max_steps)
            # 同时打印一个“中点”处 Q 值，帮助你观察策略倾向
            q_mid_L = qf.q_value(0.5, LEFT)
            q_mid_R = qf.q_value(0.5, RIGHT)
            print(f"episode={ep:>6} | greedy_success_rate={sr:.3f} | Q(0.5,L)={q_mid_L:.3f} Q(0.5,R)={q_mid_R:.3f}")

    return qf


if __name__ == "__main__":
    env = ContinuousLineWorld(start_x=0.5, step=0.05, noise_std=0.01, seed=42)
    tile_coder = TileCoder1D(num_tilings=8, tiles_per_tiling=20)

    # 演示：同一个 x 在不同动作下激活的是“不同参数块”
    demo_x = 0.63
    state_feats = tile_coder.active_features(demo_x)
    F = tile_coder.n_state_features
    left_feats = [LEFT * F + f for f in state_feats]
    right_feats = [RIGHT * F + f for f in state_feats]
    print(f"Demo x={demo_x}: state_feats={state_feats}")
    print(f"  LEFT  active_sa_feats={left_feats[:5]}... (count={len(left_feats)})")
    print(f"  RIGHT active_sa_feats={right_feats[:5]}... (count={len(right_feats)})\n")

    qf = train_sarsa_tile_coding(
        env=env,
        tile_coder=tile_coder,
        gamma=0.99,
        alpha=0.2,
        epsilon=0.1,
        n_episodes=20000,
        max_steps=300,
        seed=7
    )

    print("\nFinal greedy success rate:", evaluate_greedy_success(env_seed=2025, qf=qf, n_trials=1000, max_steps=300))
