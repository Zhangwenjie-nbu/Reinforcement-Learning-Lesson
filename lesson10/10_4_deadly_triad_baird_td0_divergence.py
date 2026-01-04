# 10_1_deadly_triad_baird_td0_divergence.py
# ------------------------------------------------------------
# 本文件做什么：
#   1) 构造一个极小的 Baird 反例（Deadly Triad 的经典发散案例变体）
#   2) 用“线性函数逼近 + TD(0)自举 + 离策略(重要性采样)”做策略评估
#   3) 观察半梯度 off-policy TD(0) 的参数 theta 如何发散（||theta|| 迅速变大）
#
# 你应该看到什么：
#   - linear off-policy TD(0) 的 ||theta||_2 会很快爆炸（发散）
#   - 对照：tabular（不使用函数逼近）通常稳定收敛到 0
# ------------------------------------------------------------

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class BairdCounterexampleEnv:
    """
    这个环境实现一个经典的“星形”反例变体：
      - 共有 6 个状态：0..4 为“上层状态”，5 为“下层状态”
      - 在上层状态：环境会确定性转移到下层状态（可以理解为没有分支选择）
      - 在下层状态：行为策略 b 以概率 p_solid 选择 solid 动作（留在下层），
                    以概率 1-p_solid 选择 dashed 动作（均匀跳到某个上层状态）
      - 目标策略 π：在下层状态永远选择 solid（因此长期停留在下层）
      - 奖励恒为 0，折扣因子 gamma 接近 1
    """

    gamma: float = 0.99
    p_solid: float = 1.0 / 6.0  # 行为策略在下层选择 solid 的概率
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.lower_state = 5
        self.upper_states = list(range(0, 5))

        # 线性特征：Phi.shape = (num_states, feature_dim)
        # 每一行就是 phi(s)^T
        # 这些特征的“刻意设计”会导致不必要的泛化，从而配合 off-policy + bootstrap 触发发散
        self.Phi = np.array(
            [
                [2, 0, 0, 0, 0, 0, 1],  # state 0
                [0, 2, 0, 0, 0, 0, 1],  # state 1
                [0, 0, 2, 0, 0, 0, 1],  # state 2
                [0, 0, 0, 2, 0, 0, 1],  # state 3
                [0, 0, 0, 0, 2, 0, 1],  # state 4
                [0, 0, 0, 0, 0, 1, 5],  # lower state 5
            ],
            dtype=float,
        )

    def reset(self) -> int:
        """随机从上层状态开始（使得行为策略能频繁覆盖上层状态）。"""
        return self.rng.choice(self.upper_states)

    def phi(self, s: int) -> np.ndarray:
        """返回状态 s 的特征向量 phi(s)，用于线性函数逼近 V(s)=phi(s)^T theta。"""
        return self.Phi[s]

    def step_behavior(self, s: int) -> Tuple[int, float, float]:
        """
        用行为策略 b 采样一步转移，并返回：
          - next_state: 下一个状态
          - reward: 奖励（本例恒为 0）
          - rho: 重要性采样比率 rho = pi(a|s) / b(a|s)

        注意：在上层状态没有真正的动作分支（可视作动作固定），因此 rho=1。
             在下层状态：目标策略永远选 solid。
               - 若行为也选 solid：rho = 1 / p_solid
               - 若行为选 dashed：rho = 0 / (1-p_solid) = 0
        """
        reward = 0.0

        if s != self.lower_state:
            # 上层 -> 下层（确定性）
            next_state = self.lower_state
            rho = 1.0
            return next_state, reward, rho

        # 下层状态：行为策略采样动作
        if self.rng.random() < self.p_solid:
            # 行为选 solid：下层 -> 下层
            next_state = self.lower_state
            rho = 1.0 / self.p_solid  # 目标策略概率为 1
            return next_state, reward, rho
        else:
            # 行为选 dashed：下层 -> 随机上层
            next_state = self.rng.choice(self.upper_states)
            rho = 0.0  # 目标策略不会选 dashed
            return next_state, reward, rho


def run_offpolicy_linear_td0(
    env: BairdCounterexampleEnv,
    steps: int = 20000,
    alpha: float = 0.01,
    theta0: np.ndarray | None = None,
    report_points: List[int] | None = None,
) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
    """
    运行半梯度 off-policy TD(0)（线性函数逼近）：
      theta <- theta + alpha * rho * delta * phi(s)
      delta = r + gamma * V(s') - V(s)

    返回：
      - theta: 训练后的参数
      - history: [(t, ||theta||_2), ...] 便于观察是否发散
    """
    if report_points is None:
        report_points = [10, 100, 1000, 5000, 10000, 20000]

    theta = np.ones(env.Phi.shape[1], dtype=float) if theta0 is None else theta0.astype(float).copy()
    s = env.reset()

    history: List[Tuple[int, float]] = []
    for t in range(1, steps + 1):
        s_next, r, rho = env.step_behavior(s)

        v_s = float(env.phi(s) @ theta)
        v_next = float(env.phi(s_next) @ theta)
        delta = r + env.gamma * v_next - v_s

        # 半梯度更新：只对 V(s) 的参数求梯度，即梯度=phi(s)
        theta += alpha * rho * delta * env.phi(s)

        if t in report_points:
            history.append((t, float(np.linalg.norm(theta, ord=2))))

        s = s_next

    return theta, history


def run_offpolicy_tabular_td0(
    env: BairdCounterexampleEnv,
    steps: int = 200000,
    alpha: float = 0.1,
    report_points: List[int] | None = None,
) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
    """
    对照实验：tabular（不使用函数逼近、不共享参数）的 off-policy TD(0)。
    在这个反例里，tabular 版本通常是稳定的（不会像线性逼近那样爆炸）。

    返回：
      - V: 每个状态一个标量值的表格
      - history: [(t, ||V||_2), ...]
    """
    if report_points is None:
        report_points = [100, 1000, 10000, 50000, 100000, 200000]

    V = np.zeros(6, dtype=float)
    s = env.reset()
    history: List[Tuple[int, float]] = []

    for t in range(1, steps + 1):
        s_next, r, rho = env.step_behavior(s)
        delta = r + env.gamma * V[s_next] - V[s]
        V[s] += alpha * rho * delta

        if t in report_points:
            history.append((t, float(np.linalg.norm(V, ord=2))))
        s = s_next

    return V, history


def main() -> None:
    """
    主入口：
      1) 跑线性 off-policy TD(0)：观察 theta 发散
      2) 跑 tabular 对照：观察稳定性
    """
    env = BairdCounterexampleEnv(gamma=0.99, p_solid=1.0 / 6.0, seed=0)

    print("=== Linear function approximation + off-policy TD(0) ===")
    theta, hist = run_offpolicy_linear_td0(env, steps=20000, alpha=0.01)
    for t, norm in hist:
        print(f"step={t:>6d} | ||theta||_2={norm:.6e}")
    print(f"final ||theta||_2 = {np.linalg.norm(theta):.6e}")

    print("\n=== Tabular (no function approximation) + off-policy TD(0) (control baseline) ===")
    V, hist_v = run_offpolicy_tabular_td0(env, steps=200000, alpha=0.1)
    for t, norm in hist_v:
        print(f"step={t:>6d} | ||V||_2={norm:.6e}")
    print(f"final V = {V}")


if __name__ == "__main__":
    main()
