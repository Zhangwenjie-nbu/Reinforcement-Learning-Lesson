# -*- coding: utf-8 -*-
"""
lesson10_6_mspbe_gtd2_tdc.py

整体在干什么？
1) 使用 Baird counterexample（致命三元组经典反例）作为 off-policy 线性评估任务：
   - 奖励恒为 0（真值很简单），但 off-policy + bootstrap + function approximation 会让 TD(0) 可能发散
2) 对比三种算法：
   (A) Off-policy linear TD(0)（semi-gradient）:      theta <- theta + alpha * rho * delta * phi
   (B) GTD2（Gradient TD 之一，近似最小化 MSPBE）:
       w     <- w + beta  * (rho*delta - phi^T w) * phi
       theta <- theta + alpha * rho * (phi - gamma*phi') * (phi^T w)
   (C) TDC（GTD 的等价变体，常见写法）:
       w     <- w + beta  * (rho*delta - phi^T w) * phi
       theta <- theta + alpha * [ rho*delta*phi - gamma*rho*phi'*(phi^T w) ]
3) 同时在线估计 A, b, C（用样本均值），每隔一段打印 MSPBE_hat(theta)：
       MSPBE_hat(theta) = (b_hat - A_hat theta)^T * C_hat^{-1} * (b_hat - A_hat theta)
   这帮助你看到 GTD2/TDC 确实在“推动某个目标函数变小”。

你需要掌握：
- MSPBE 的二次型形式以及 A,b,C 的含义
- GTD2/TDC 为什么需要辅助变量 w：近似 C^{-1}(b - A theta)
- 两时间尺度：beta 通常比 alpha 大（w 追得更快）
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class BairdCounterexampleEnv:
    """
    Baird 反例（6 状态）：
      - 0..4：上层状态
      - 5：下层状态
    行为策略 b（在下层）：
      - solid: 以 p_solid 留在下层
      - dashed: 以 1-p_solid 跳到随机上层
    目标策略 π（在下层）：
      - 永远选 solid
    上层状态到下层是确定性转移（视作 rho=1）。

    奖励恒为 0（便于观察：真值应为 0，但 TD(0) 仍可能发散）。
    """

    gamma: float = 0.99
    p_solid: float = 1.0 / 6.0
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.lower_state = 5
        self.upper_states = list(range(0, 5))

        # 线性特征矩阵 Phi: shape = (6, 7)
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
        """从随机上层状态开始（保证行为分布频繁覆盖上层）。"""
        return self.rng.choice(self.upper_states)

    def phi(self, s: int) -> np.ndarray:
        """返回 phi(s)。"""
        return self.Phi[s]

    def step_behavior(self, s: int) -> Tuple[int, float, float]:
        """
        用行为策略 b 采样一步，并返回：
          next_state, reward, rho = pi(a|s)/b(a|s)

        在上层：无分支，rho=1，上层->下层
        在下层：
          - 若 b 选 solid：rho = 1/p_solid，下层->下层
          - 若 b 选 dashed：rho = 0，下层->随机上层
        """
        reward = 0.0

        if s != self.lower_state:
            return self.lower_state, reward, 1.0

        if self.rng.random() < self.p_solid:
            return self.lower_state, reward, 1.0 / self.p_solid
        else:
            return self.rng.choice(self.upper_states), reward, 0.0


def estimate_mspbe(A_hat: np.ndarray, b_hat: np.ndarray, C_hat: np.ndarray, theta: np.ndarray, ridge: float = 1e-6) -> float:
    """
    计算 MSPBE_hat(theta) = (b - A theta)^T C^{-1} (b - A theta) 的样本估计。
    为避免 C 奇异，加一个 ridge。
    """
    d = theta.shape[0]
    C_reg = C_hat + ridge * np.eye(d)
    res = b_hat - A_hat @ theta
    x = np.linalg.solve(C_reg, res)
    return float(res.T @ x)


def update_abc_running(
    A_sum: np.ndarray,
    b_sum: np.ndarray,
    C_sum: np.ndarray,
    count: int,
    phi: np.ndarray,
    phi_next: np.ndarray,
    r: float,
    rho: float,
    gamma: float,
) -> int:
    """
    用样本均值估计 A,b,C：
      A = E[ rho * phi (phi - gamma phi_next)^T ]
      b = E[ rho * r * phi ]
      C = E[ phi phi^T ]
    这里用累加和 + count 形成均值。
    """
    A_sum += rho * np.outer(phi, (phi - gamma * phi_next))
    b_sum += rho * r * phi
    C_sum += np.outer(phi, phi)
    return count + 1


def run_td0(env: BairdCounterexampleEnv, steps: int, alpha: float, seed: int = 0, report_every: int = 2000):
    """
    Off-policy linear TD(0)（semi-gradient）：
      delta = r + gamma*theta^T phi' - theta^T phi
      theta <- theta + alpha * rho * delta * phi
    同时估计 MSPBE_hat 用于观察。
    """
    rng = random.Random(seed)
    theta = np.ones(env.Phi.shape[1], dtype=float)
    s = env.reset()

    d = theta.shape[0]
    A_sum = np.zeros((d, d), dtype=float)
    b_sum = np.zeros(d, dtype=float)
    C_sum = np.zeros((d, d), dtype=float)
    cnt = 0

    history = []

    for t in range(1, steps + 1):
        s_next, r, rho = env.step_behavior(s)
        phi = env.phi(s)
        phi_next = env.phi(s_next)

        v = float(phi @ theta)
        v_next = float(phi_next @ theta)
        delta = r + env.gamma * v_next - v

        theta += alpha * rho * delta * phi

        cnt = update_abc_running(A_sum, b_sum, C_sum, cnt, phi, phi_next, r, rho, env.gamma)

        if t % report_every == 0:
            A_hat = A_sum / cnt
            b_hat = b_sum / cnt
            C_hat = C_sum / cnt
            mspbe = estimate_mspbe(A_hat, b_hat, C_hat, theta)
            history.append((t, float(np.linalg.norm(theta)), mspbe))

        s = s_next

    return theta, history


def run_gtd2(env: BairdCounterexampleEnv, steps: int, alpha: float, beta: float, seed: int = 0, report_every: int = 2000):
    """
    GTD2（近似对 MSPBE 做随机梯度下降）：
      delta = r + gamma*theta^T phi' - theta^T phi
      w     <- w + beta * (rho*delta - phi^T w) * phi
      theta <- theta + alpha * rho * (phi - gamma*phi') * (phi^T w)
    """
    rng = random.Random(seed)
    d = env.Phi.shape[1]
    theta = np.ones(d, dtype=float)
    w = np.zeros(d, dtype=float)

    s = env.reset()

    A_sum = np.zeros((d, d), dtype=float)
    b_sum = np.zeros(d, dtype=float)
    C_sum = np.zeros((d, d), dtype=float)
    cnt = 0

    history = []

    for t in range(1, steps + 1):
        s_next, r, rho = env.step_behavior(s)
        phi = env.phi(s)
        phi_next = env.phi(s_next)

        v = float(phi @ theta)
        v_next = float(phi_next @ theta)
        delta = r + env.gamma * v_next - v

        # w update (fast time-scale)
        w += beta * (rho * delta - float(phi @ w)) * phi

        # theta update (slow time-scale)
        theta += alpha * rho * (phi - env.gamma * phi_next) * float(phi @ w)

        cnt = update_abc_running(A_sum, b_sum, C_sum, cnt, phi, phi_next, r, rho, env.gamma)

        if t % report_every == 0:
            A_hat = A_sum / cnt
            b_hat = b_sum / cnt
            C_hat = C_sum / cnt
            mspbe = estimate_mspbe(A_hat, b_hat, C_hat, theta)
            history.append((t, float(np.linalg.norm(theta)), mspbe, float(np.linalg.norm(w))))

        s = s_next

    return theta, w, history


def run_tdc(env: BairdCounterexampleEnv, steps: int, alpha: float, beta: float, seed: int = 0, report_every: int = 2000):
    """
    TDC（GTD 的常见变体）：
      delta = r + gamma*theta^T phi' - theta^T phi
      w     <- w + beta * (rho*delta - phi^T w) * phi
      theta <- theta + alpha * [ rho*delta*phi - gamma*rho*phi'*(phi^T w) ]
    """
    rng = random.Random(seed)
    d = env.Phi.shape[1]
    theta = np.ones(d, dtype=float)
    w = np.zeros(d, dtype=float)

    s = env.reset()

    A_sum = np.zeros((d, d), dtype=float)
    b_sum = np.zeros(d, dtype=float)
    C_sum = np.zeros((d, d), dtype=float)
    cnt = 0

    history = []

    for t in range(1, steps + 1):
        s_next, r, rho = env.step_behavior(s)
        phi = env.phi(s)
        phi_next = env.phi(s_next)

        v = float(phi @ theta)
        v_next = float(phi_next @ theta)
        delta = r + env.gamma * v_next - v

        # w update
        w += beta * (rho * delta - float(phi @ w)) * phi

        # theta update
        theta += alpha * (rho * delta * phi - env.gamma * rho * phi_next * float(phi @ w))

        cnt = update_abc_running(A_sum, b_sum, C_sum, cnt, phi, phi_next, r, rho, env.gamma)

        if t % report_every == 0:
            A_hat = A_sum / cnt
            b_hat = b_sum / cnt
            C_hat = C_sum / cnt
            mspbe = estimate_mspbe(A_hat, b_hat, C_hat, theta)
            history.append((t, float(np.linalg.norm(theta)), mspbe, float(np.linalg.norm(w))))

        s = s_next

    return theta, w, history


def main() -> None:
    """
    主入口：在同一环境上跑 TD(0)、GTD2、TDC，并打印对比。
    """
    env = BairdCounterexampleEnv(gamma=0.99, p_solid=1.0 / 6.0, seed=0)

    steps = 40000
    report_every = 2000

    print("=== (A) Off-policy linear TD(0) (often diverges) ===")
    theta_td, hist_td = run_td0(env, steps=steps, alpha=0.01, report_every=report_every)
    for t, norm_th, mspbe in hist_td:
        print(f"step={t:>6} | ||theta||={norm_th:>10.4e} | MSPBE_hat={mspbe:>10.4e}")
    print(f"final ||theta|| = {np.linalg.norm(theta_td):.4e}\n")

    print("=== (B) GTD2 (should be stable; minimizes MSPBE) ===")
    theta_g, w_g, hist_g = run_gtd2(env, steps=steps, alpha=0.005, beta=0.05, report_every=report_every)
    for t, norm_th, mspbe, norm_w in hist_g:
        print(f"step={t:>6} | ||theta||={norm_th:>10.4e} | ||w||={norm_w:>10.4e} | MSPBE_hat={mspbe:>10.4e}")
    print(f"final ||theta|| = {np.linalg.norm(theta_g):.4e} | final ||w|| = {np.linalg.norm(w_g):.4e}\n")

    print("=== (C) TDC (often comparable stability) ===")
    theta_t, w_t, hist_t = run_tdc(env, steps=steps, alpha=0.005, beta=0.05, report_every=report_every)
    for t, norm_th, mspbe, norm_w in hist_t:
        print(f"step={t:>6} | ||theta||={norm_th:>10.4e} | ||w||={norm_w:>10.4e} | MSPBE_hat={mspbe:>10.4e}")
    print(f"final ||theta|| = {np.linalg.norm(theta_t):.4e} | final ||w|| = {np.linalg.norm(w_t):.4e}\n")


if __name__ == "__main__":
    main()
