# -*- coding: utf-8 -*-
"""
lesson4_2_gamma_effects.py

整体在干什么？
通过三个小实验展示折扣因子 gamma 的两层意义（偏好 + 数学稳定性直觉）：

Experiment A: 有效视野（effective horizon）
- 对给定阈值 epsilon（如0.01），计算需要多少步 k 才使 gamma^k <= epsilon。
- gamma 越接近 1，k 越大，意味着你需要考虑更远未来。

Experiment B: 同一条 reward 序列的回报对比
- 固定一条 rewards 序列，分别用不同 gamma 计算 G0。
- gamma 越大，越“看重后面的奖励”。

Experiment C: 回报估计的波动（方差直觉）
- 生成大量带噪的 rewards 序列（模拟随机环境的样本轨迹）
- 对每条序列计算 G0，比较不同 gamma 下 G0 的样本方差
- gamma 越大，通常方差越大（更难稳定估计），对应“更长视野、更慢收敛”的工程事实。

你需要掌握：
- gamma 影响权重衰减速度（偏好/视野）
- gamma 越接近 1，值函数上界 Rmax/(1-gamma) 越大，误差传播更强
"""

import math
import random


def effective_horizon(gamma: float, epsilon: float = 0.01) -> int:
    """
    计算最小整数 k，使得 gamma^k <= epsilon。
    这给出一个“权重衰减到可忽略”的步数尺度。

    逻辑：
    - 若 gamma=0，则 k=1（0^1=0 <= epsilon）
    - 若 gamma 接近 1，则 k 会很大
    """
    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must be in [0, 1).")
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must be in (0, 1).")

    if gamma == 0.0:
        return 1

    k = math.log(epsilon) / math.log(gamma)
    return math.ceil(k)


def discounted_return(rewards, gamma: float) -> float:
    """
    计算折扣回报 G0 = r1 + gamma*r2 + gamma^2*r3 + ...
    rewards[k] 对应 R_{k+1}
    """
    G = 0.0
    power = 1.0
    for r in rewards:
        G += power * r
        power *= gamma
    return G


def generate_noisy_rewards(rng: random.Random, length: int = 50, noise_std: float = 1.0):
    """
    生成一条带噪 rewards 序列（用于方差实验）：
    - 均值为 0 的高斯噪声奖励
    - 这代表“没有明显奖励趋势”的随机环境样本
    """
    return [rng.gauss(0.0, noise_std) for _ in range(length)]


def sample_return_variance(gammas, n_samples: int = 5000, length: int = 50, noise_std: float = 1.0, seed: int = 0):
    """
    对每个 gamma：
    - 采样 n_samples 条 rewards 序列
    - 计算每条的 G0
    - 返回样本均值与样本方差

    逻辑：
    - 用来展示：gamma 越大，G0 通常波动越大（更难估计）
    """
    rng = random.Random(seed)
    stats = {}

    for gamma in gammas:
        Gs = []
        for _ in range(n_samples):
            rewards = generate_noisy_rewards(rng, length=length, noise_std=noise_std)
            Gs.append(discounted_return(rewards, gamma))

        mu = sum(Gs) / len(Gs)
        if len(Gs) < 2:
            var = 0.0
        else:
            sse = sum((g - mu) ** 2 for g in Gs)
            var = sse / (len(Gs) - 1)

        stats[gamma] = (mu, var)

    return stats


if __name__ == "__main__":
    gammas = [0.5, 0.9, 0.95, 0.99]

    print("=== Experiment A: Effective horizon (gamma^k <= 0.01) ===")
    for g in gammas:
        k = effective_horizon(g, epsilon=0.01)
        print(f"gamma={g:>4} -> k ≈ {k} steps")

    print("\n=== Experiment B: Same reward sequence, different gamma ===")
    rewards_demo = [0.0] * 10 + [1.0]  # 前10步无奖励，第11步才出现+1
    print("rewards:", rewards_demo)
    for g in gammas:
        G0 = discounted_return(rewards_demo, g)
        print(f"gamma={g:>4} -> G0={G0:.6f}")

    print("\n=== Experiment C: Variance of G0 under noisy rewards ===")
    stats = sample_return_variance(gammas, n_samples=8000, length=60, noise_std=1.0, seed=42)
    print("gamma | mean(G0)   var(G0)")
    print("------+-----------------------")
    for g in gammas:
        mu, var = stats[g]
        print(f"{g:>4} | {mu:>8.4f}  {var:>10.4f}")
