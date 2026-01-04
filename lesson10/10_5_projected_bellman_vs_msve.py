# -*- coding: utf-8 -*-
"""
lesson10_5_projected_bellman_vs_msve.py

整体在干什么？
1) 构造一个小型 MRP（3 个状态），给出转移矩阵 P、奖励向量 r、折扣 gamma。
2) 定义一个“有表达能力缺陷”的线性特征矩阵 Phi（3x2），使得无法精确表示真实 v。
3) 计算：
   - 真实价值 v = (I - gamma P)^(-1) r
   - 最小 MSVE（D-加权最小二乘）解 theta_LS
   - 投影贝尔曼方程（PBE）固定点解 theta_TD（满足 A theta = b）
4) 用采样仿真跑 TD(0)，验证它会收敛到 theta_TD，而不是 theta_LS。

你需要掌握：
- theta_LS 与 theta_TD 一般不同：TD(0)不是在最小化 MSVE
- TD(0)的固定点对应“投影贝尔曼不动点”：Phi theta = Pi T(Phi theta)
"""

from __future__ import annotations

import numpy as np


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    计算马尔可夫链的平稳分布 d，使得 d^T P = d^T 且 sum(d)=1。
    用线性方程求解（加一条归一化约束）。
    """
    n = P.shape[0]
    A = (P.T - np.eye(n))
    A = np.vstack([A, np.ones((1, n))])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    # 最小二乘解（在数值上更稳）
    d, *_ = np.linalg.lstsq(A, b, rcond=None)
    d = np.clip(d, 0.0, None)
    d = d / d.sum()
    return d


def true_value(P: np.ndarray, r: np.ndarray, gamma: float) -> np.ndarray:
    """
    计算真实价值函数 v = (I - gamma P)^(-1) r。
    """
    n = P.shape[0]
    v = np.linalg.solve(np.eye(n) - gamma * P, r)
    return v


def theta_ls(Phi: np.ndarray, D: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    计算最小 MSVE 的最小二乘解：
      theta_LS = argmin ||Phi theta - v||_D^2
    对应正规方程：
      (Phi^T D Phi) theta = Phi^T D v
    """
    A = Phi.T @ D @ Phi
    b = Phi.T @ D @ v
    return np.linalg.solve(A, b)


def theta_td(Phi: np.ndarray, D: np.ndarray, P: np.ndarray, r: np.ndarray, gamma: float) -> np.ndarray:
    """
    计算线性 on-policy TD(0) 的固定点解（PBE 的代数形式）：
      A theta = b
    其中：
      A = Phi^T D (I - gamma P) Phi
      b = Phi^T D r
    """
    n = P.shape[0]
    A = Phi.T @ D @ (np.eye(n) - gamma * P) @ Phi
    b = Phi.T @ D @ r
    return np.linalg.solve(A, b)


def td0_simulation(P: np.ndarray, r: np.ndarray, Phi: np.ndarray, gamma: float, alpha: float, steps: int, seed: int = 0) -> np.ndarray:
    """
    用采样的方式运行 TD(0)（on-policy）来学习 theta：
      delta = r(s) + gamma * v_hat(s') - v_hat(s)
      theta <- theta + alpha * delta * phi(s)

    说明：
    - 这里把 r 写成状态奖励 r(s)，转移由 P 采样得到 s'。
    - 这会收敛到 theta_TD（在常见条件下、足够小步长/足够长时间）。
    """
    rng = np.random.default_rng(seed)
    n, d = Phi.shape
    theta = np.zeros(d)

    # 从一个随机状态开始
    s = int(rng.integers(0, n))

    for _ in range(steps):
        # 按 P 采样下一状态
        s_next = int(rng.choice(n, p=P[s]))

        v_s = float(Phi[s] @ theta)
        v_next = float(Phi[s_next] @ theta)
        delta = float(r[s] + gamma * v_next - v_s)

        theta = theta + alpha * delta * Phi[s]

        s = s_next

    return theta


def msve(Phi: np.ndarray, D: np.ndarray, v_true: np.ndarray, theta: np.ndarray) -> float:
    """
    计算 MSVE(theta) = ||Phi theta - v_true||_D^2
    """
    err = Phi @ theta - v_true
    return float(err.T @ D @ err)


def main() -> None:
    # ---------- 1) 构造一个 3-state MRP ----------
    P = np.array(
        [
            [0.50, 0.50, 0.00],
            [0.00, 0.50, 0.50],
            [0.50, 0.00, 0.50],
        ],
        dtype=float,
    )
    r = np.array([0.0, 0.0, 1.0], dtype=float)  # 只有状态2有奖励
    gamma = 0.9

    # ---------- 2) 特征：3x2，刻意让 state0/state1 在特征上“不可区分” ----------
    # Phi[s] = [bias, indicator(state==2)]
    # 这意味着 state0 与 state1 永远预测相同价值，无法精确拟合真实 v。
    Phi = np.array(
        [
            [1.0, 0.0],  # state 0
            [1.0, 0.0],  # state 1
            [1.0, 1.0],  # state 2
        ],
        dtype=float,
    )

    # ---------- 3) 平稳分布与真实价值 ----------
    d = stationary_distribution(P)
    D = np.diag(d)
    v = true_value(P, r, gamma)

    # ---------- 4) 两个不同“目标”下的解 ----------
    th_ls = theta_ls(Phi, D, v)
    th_td = theta_td(Phi, D, P, r, gamma)

    v_hat_ls = Phi @ th_ls
    v_hat_td = Phi @ th_td

    # ---------- 5) 用采样 TD(0) 验证收敛到 theta_TD ----------
    th_sim = td0_simulation(P, r, Phi, gamma=gamma, alpha=0.02, steps=200000, seed=7)
    v_hat_sim = Phi @ th_sim

    # ---------- 6) 输出对比 ----------
    print("=== MRP definition ===")
    print("P=\n", P)
    print("r=", r)
    print("gamma=", gamma)
    print("stationary d=", np.round(d, 6))
    print()

    print("=== True value v ===")
    print("v_true=", np.round(v, 6))
    print()

    print("=== Least-squares (min MSVE) solution ===")
    print("theta_LS=", np.round(th_ls, 6))
    print("v_hat_LS=", np.round(v_hat_ls, 6))
    print("MSVE(L S)=", msve(Phi, D, v, th_ls))
    print()

    print("=== TD fixed point (Projected Bellman) solution ===")
    print("theta_TD=", np.round(th_td, 6))
    print("v_hat_TD=", np.round(v_hat_td, 6))
    print("MSVE(T D)=", msve(Phi, D, v, th_td))
    print()

    print("=== TD(0) simulation result (should be close to theta_TD) ===")
    print("theta_sim=", np.round(th_sim, 6))
    print("v_hat_sim=", np.round(v_hat_sim, 6))
    print("MSVE(sim)=", msve(Phi, D, v, th_sim))
    print()

    print("=== Key takeaway ===")
    print("theta_TD equals theta_LS? ->", np.allclose(th_td, th_ls, atol=1e-4))


if __name__ == "__main__":
    main()
