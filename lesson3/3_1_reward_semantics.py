# -*- coding: utf-8 -*-
"""
lesson3_1_reward_semantics.py

整体在干什么？
1) 构造一个极简 MDP，用于演示“奖励机制”与“奖励样本”的区别：
   - 转移 S_{t+1} 具有随机性：P(s'|s,a)
   - 奖励由确定性部分 R(s,a,s') + 零均值噪声 组成
2) 展示：同一个 (s,a) 下，多次交互得到的奖励样本 R_{t+1} 会变化（随机变量的实现）。
3) 用采样估计：
   - r(s,a) = E[R_{t+1} | S_t=s, A_t=a]
   - r(s,a,s') = E[R_{t+1} | S_t=s, A_t=a, S_{t+1}=s']
4) 同时给出“理论真值”（因为我们人为设定了转移概率与奖励规则），方便你核对。

你需要从这份代码掌握的概念：
- R(s,a,s')：奖励规则/确定性部分（或更一般的奖励分布机制）
- R_{t+1}：一次交互实际拿到的奖励样本
- r(s,a)、r(s,a,s')：对未来不确定性的正确处理方式（条件期望）
"""

import random

A0, A1 = 0, 1  # 两个动作


class TinyMDP:
    """
    一个极简 MDP：
    - 状态：0 或 1
    - 动作：0 或 1
    - 转移：从 s=0 出发，根据动作决定到 s'=1 的概率
    - 奖励：确定性规则 base_reward(s,a,s') + 高斯噪声（均值0）
    """

    def __init__(self, noise_std: float = 0.3, seed: int = 0):
        self.noise_std = noise_std
        self.rng = random.Random(seed)

        # 从 s=0 出发，a=0 更容易到 s'=1；a=1 次之
        self.p_to_1 = {
            (0, A0): 0.7,
            (0, A1): 0.4,
            # 从 s=1 出发，为了简单设为“保持在1”（本节不讨论长期）
            (1, A0): 1.0,
            (1, A1): 1.0,
        }

    def base_reward(self, s: int, a: int, s2: int) -> float:
        """
        确定性奖励规则 R(s,a,s') 的“确定性部分”：
        - 如果下一状态到达 1，则给 +1，否则 0
        - 动作 a=1 有一个固定成本 0.1（演示动作相关奖励）
        """
        reach_bonus = 1.0 if s2 == 1 else 0.0
        action_cost = 0.1 if a == A1 else 0.0
        return reach_bonus - action_cost

    def step(self, s: int, a: int):
        """
        执行一步环境交互，返回 (s2, r)：
        1) 先按 P(s'|s,a) 采样得到下一状态 s2
        2) 再按奖励机制生成一次奖励样本 r = R(s,a,s2) + noise

        这体现了关键点：
        - 你看到的 r 是 R_{t+1} 的一个样本，而不是“规则本身”
        """
        p = self.p_to_1[(s, a)]
        s2 = 1 if self.rng.random() < p else 0

        noise = self.rng.gauss(0.0, self.noise_std)
        r = self.base_reward(s, a, s2) + noise
        return s2, r

    def true_r_s_a_s2(self, s: int, a: int, s2: int) -> float:
        """
        理论上的 r(s,a,s') = E[R_{t+1} | s,a,s']。
        因为噪声均值为0，所以等于 base_reward(s,a,s')
        """
        return self.base_reward(s, a, s2)

    def true_r_s_a(self, s: int, a: int) -> float:
        """
        理论上的 r(s,a) = E[R_{t+1} | s,a]
        = sum_{s'} P(s'|s,a) * E[R_{t+1} | s,a,s']
        """
        p1 = self.p_to_1[(s, a)]
        p0 = 1.0 - p1
        return p0 * self.true_r_s_a_s2(s, a, 0) + p1 * self.true_r_s_a_s2(s, a, 1)


def estimate_r_s_a(env: TinyMDP, s: int, a: int, n: int = 20000) -> float:
    """
    用采样估计 r(s,a)：
    - 固定 (s,a)，重复 n 次 step
    - 对奖励样本求平均，近似 E[R_{t+1}|s,a]
    """
    total = 0.0
    for _ in range(n):
        _, r = env.step(s, a)
        total += r
    return total / n


def estimate_r_s_a_s2(env: TinyMDP, s: int, a: int, target_s2: int, n: int = 200000) -> float:
    """
    用采样估计 r(s,a,s')：
    - 固定 (s,a)，不断采样 (s2,r)
    - 只保留 s2 == target_s2 的样本来做条件期望估计

    注意：
    - 如果 P(s'=target|s,a) 很小，需要更大 n 才能收集到足够样本。
    - 这正对应“条件事件稀有时，估计更困难”的统计事实。
    """
    total = 0.0
    cnt = 0
    for _ in range(n):
        s2, r = env.step(s, a)
        if s2 == target_s2:
            total += r
            cnt += 1
    return (total / cnt) if cnt > 0 else float("nan")


if __name__ == "__main__":
    env = TinyMDP(noise_std=0.3, seed=42)

    s = 0
    a = A0

    print("=== (1) 同一个 (s,a) 下，奖励样本 R_{t+1} 会变化 ===")
    for i in range(5):
        s2, r = env.step(s, a)
        print(f"trial {i}: s={s}, a={a} -> s'={s2}, reward_sample={r:.4f}")

    print("\n=== (2) 估计 r(s,a) = E[R_{t+1}|s,a] 并对照理论真值 ===")
    est = estimate_r_s_a(env, s=0, a=A0, n=30000)
    truth = env.true_r_s_a(s=0, a=A0)
    print(f"estimate r(0,A0) = {est:.4f}")
    print(f"true     r(0,A0) = {truth:.4f}")

    est2 = estimate_r_s_a(env, s=0, a=A1, n=30000)
    truth2 = env.true_r_s_a(s=0, a=A1)
    print(f"estimate r(0,A1) = {est2:.4f}")
    print(f"true     r(0,A1) = {truth2:.4f}")

    print("\n=== (3) 估计 r(s,a,s') = E[R_{t+1}|s,a,s'] 并对照理论真值 ===")
    est_c1 = estimate_r_s_a_s2(env, s=0, a=A0, target_s2=1, n=200000)
    truth_c1 = env.true_r_s_a_s2(s=0, a=A0, s2=1)
    print(f"estimate r(0,A0, s'=1) = {est_c1:.4f} | true = {truth_c1:.4f}")

    est_c0 = estimate_r_s_a_s2(env, s=0, a=A0, target_s2=0, n=200000)
    truth_c0 = env.true_r_s_a_s2(s=0, a=A0, s2=0)
    print(f"estimate r(0,A0, s'=0) = {est_c0:.4f} | true = {truth_c0:.4f}")
