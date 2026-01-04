# -*- coding: utf-8 -*-
"""
lesson10_2_tile_coding_td0.py

整体在干什么？
1) 构造一个连续状态环境 ContinuousLineWorld：
   - 状态 x ∈ [0,1]
   - 动作 LEFT/RIGHT
   - 到达 x=1 终止并奖励 +1；到达 x=0 终止奖励 0
2) 实现 1D Tile Coding（多组错位 tilings）：
   - 每个 tiling 把 [0,1] 切成 N 个 tile
   - 不同 tiling 采用不同 offset（平移），实现更细分辨率
   - 对任意 x，返回 M 个激活特征（稀疏）
3) 用线性 TD(0)（semi-gradient）学习价值函数 V(x)：
      V(x) = sum_{i in active(x)} w[i]
      delta = r + gamma*V(x') - V(x)
      对每个激活特征 i：w[i] += (alpha/M) * delta
4) 打印：
   - 某个示例 x 的激活特征索引（帮助你直观看到“稀疏激活”）
   - 训练后在若干 x 网格点上的 V(x) 估计（应随 x 增大而增大）

你需要掌握：
- Tile coding 的核心是：多 tilings + offset + 稀疏二值特征
- 它带来局部泛化：更新只影响附近区域，而不是全局
- alpha 常按 tilings 数缩放：alpha/M
"""

import random
from dataclasses import dataclass
from typing import List
import math


LEFT, RIGHT = 0, 1


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
        return self.start_x

    def is_terminal(self, x: float) -> bool:
        return x <= 0.0 or x >= 1.0

    def step_env(self, x: float, a: int):
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


class UniformRandomPolicy:
    """均匀随机策略：LEFT/RIGHT 等概率。"""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, x: float) -> int:
        return self.rng.choice([LEFT, RIGHT])


@dataclass
class TileCoder1D:
    """
    1D Tile Coder（不使用 hashing，直接显式索引，便于理解）
    - num_tilings: tilings 数 M
    - tiles_per_tiling: 每个 tiling 的 tile 数 N
    特征总数 = M * N
    """

    num_tilings: int
    tiles_per_tiling: int
    x_min: float = 0.0
    x_max: float = 1.0

    def __post_init__(self):
        self.width = (self.x_max - self.x_min) / self.tiles_per_tiling

    def active_features(self, x: float) -> List[int]:
        """
        给定 x，返回所有激活特征的索引（长度= num_tilings）
        关键点：
        - 每个 tiling 有一个不同 offset（平移）
        - 计算 tile_index 后映射到全局特征 id：tiling_id * N + tile_index
        """
        # 避免边界 x==x_max 时落到 tile=N 的越界
        x = min(max(x, self.x_min), self.x_max - 1e-12)

        feats = []
        for t in range(self.num_tilings):
            # 让 offset 均匀分布在 [0, width) 之间
            offset = (t / self.num_tilings) * self.width
            scaled = (x - self.x_min + offset) / self.width
            tile_index = int(math.floor(scaled))
            if tile_index < 0:
                tile_index = 0
            if tile_index >= self.tiles_per_tiling:
                tile_index = self.tiles_per_tiling - 1

            feat_id = t * self.tiles_per_tiling + tile_index
            feats.append(feat_id)

        return feats

    @property
    def n_features(self) -> int:
        return self.num_tilings * self.tiles_per_tiling


@dataclass
class TileValueFunction:
    """
    基于 tile coding 的线性价值函数：
    V(x) = sum_{i in active(x)} w[i]
    """

    tile_coder: TileCoder1D
    w: List[float]

    def value(self, x: float) -> float:
        feats = self.tile_coder.active_features(x)
        return sum(self.w[i] for i in feats)

    def update(self, x: float, delta: float, alpha: float):
        feats = self.tile_coder.active_features(x)
        step = alpha / self.tile_coder.num_tilings  # 常用缩放
        for i in feats:
            self.w[i] += step * delta


def train_linear_td0_with_tile_coding(
    env: ContinuousLineWorld,
    policy: UniformRandomPolicy,
    tile_coder: TileCoder1D,
    gamma=0.99,
    alpha=0.1,
    n_episodes=5000,
    max_steps=300,
):
    """
    用 TD(0) 做策略评估（on-policy）：
    - 采样 (x, a, r, x')
    - delta = r + gamma*V(x') - V(x)
    - w <- w + (alpha/M)*delta * 1_{active}
    """
    vf = TileValueFunction(tile_coder=tile_coder, w=[0.0] * tile_coder.n_features)

    checkpoints = [10, 50, 200, 1000, n_episodes]

    for ep in range(1, n_episodes + 1):
        x = env.reset()

        for _ in range(max_steps):
            a = policy.act(x)
            x2, r, done = env.step_env(x, a)

            if env.is_terminal(x):
                break

            v_x = vf.value(x)
            v_x2 = 0.0 if env.is_terminal(x2) else vf.value(x2)

            delta = r + gamma * v_x2 - v_x
            vf.update(x, delta, alpha)

            x = x2
            if done:
                break

        if ep in checkpoints:
            grid = [i / 10 for i in range(11)]
            vals = [round(vf.value(g), 3) for g in grid]
            print(f"episode={ep:>5} | V(grid 0.0..1.0 step0.1) = {vals}")

    return vf


if __name__ == "__main__":
    env = ContinuousLineWorld(start_x=0.5, step=0.05, noise_std=0.01, seed=42)
    policy = UniformRandomPolicy(seed=7)

    tile_coder = TileCoder1D(num_tilings=8, tiles_per_tiling=20, x_min=0.0, x_max=1.0)

    # 展示“稀疏激活”：一个 x 只激活 M 个特征
    x_demo = 0.63
    feats = tile_coder.active_features(x_demo)
    print(f"Demo x={x_demo}: active feature ids = {feats} (count={len(feats)})")
    print(f"Total features = {tile_coder.n_features}\n")

    vf = train_linear_td0_with_tile_coding(
        env=env,
        policy=policy,
        tile_coder=tile_coder,
        gamma=0.99,
        alpha=0.1,
        n_episodes=5000,
        max_steps=300
    )

    print("\nFinal V(x) on grid:")
    for g in [i / 10 for i in range(11)]:
        print(f"x={g:.1f} -> V={vf.value(g):.4f}")
