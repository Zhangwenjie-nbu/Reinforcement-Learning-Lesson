# -*- coding: utf-8 -*-
"""
lesson4_1_return_recursion.py

整体在干什么？
1) 随机生成若干条 episode 的奖励序列 rewards = [R_1, R_2, ..., R_T]（长度T可变，模拟终止）。
2) 用两种方式计算回报 G_0：
   (A) 按定义求和：G_0 = R_1 + gamma*R_2 + gamma^2*R_3 + ...
   (B) 按递推计算：G_T = 0，G_{t} = R_{t+1} + gamma*G_{t+1}（从后往前算）
3) 对比两种计算结果，证明它们数值一致（误差仅来自浮点运算）。

你需要从这份代码掌握的概念：
- “递推形式”是“求和定义”的等价重写，不是额外假设
- 终止对应的边界条件是 G_T = 0
"""

import random


def discounted_return_by_sum(rewards, gamma: float) -> float:
    """
    用“求和定义”计算 G_0：
    G_0 = sum_{k=0}^{T-1} gamma^k * rewards[k]
    其中 rewards[k] 表示 R_{k+1}
    """
    G = 0.0
    power = 1.0
    for r in rewards:
        G += power * r
        power *= gamma
    return G


def discounted_return_by_recursion(rewards, gamma: float) -> float:
    """
    用“递推定义”计算 G_0：
    令 G_T = 0，然后从后往前：
    G_{t} = R_{t+1} + gamma * G_{t+1}

    这里 rewards = [R_1, ..., R_T]
    所以从最后一个奖励开始反推即可。
    """
    G_next = 0.0  # 对应 G_T
    for r in reversed(rewards):
        G_next = r + gamma * G_next
    return G_next  # 此时就是 G_0


def generate_random_episode_rewards(rng: random.Random, max_len: int = 20, terminate_prob: float = 0.15):
    """
    随机生成一条 episode 的奖励序列（用于演示）：
    - 每一步奖励是一个带噪声的实数（这里用均匀分布）
    - 每一步以 terminate_prob 概率“终止”，从而得到可变长度 T
    - 最长不超过 max_len，避免无限循环
    """
    rewards = []
    for _ in range(max_len):
        # 生成一步奖励（可正可负，模拟一般环境中的即时反馈）
        r = rng.uniform(-1.0, 1.0)
        rewards.append(r)

        # 随机终止
        if rng.random() < terminate_prob:
            break
    return rewards


def run_demo(gamma: float = 0.95, n_episodes: int = 5, seed: int = 42):
    """
    运行演示：
    - 打印若干条 episode 的 rewards
    - 对每条 episode 计算两种 G_0 并比较差异
    """
    rng = random.Random(seed)

    for i in range(n_episodes):
        rewards = generate_random_episode_rewards(rng)
        g_sum = discounted_return_by_sum(rewards, gamma)
        g_rec = discounted_return_by_recursion(rewards, gamma)
        diff = abs(g_sum - g_rec)

        print(f"\nEpisode {i}: T={len(rewards)}")
        print("rewards:", [round(r, 4) for r in rewards])
        print(f"G0 by sum      = {g_sum:.10f}")
        print(f"G0 by recursion= {g_rec:.10f}")
        print(f"|difference|   = {diff:.12e}")


if __name__ == "__main__":
    run_demo(gamma=0.95, n_episodes=6, seed=7)
