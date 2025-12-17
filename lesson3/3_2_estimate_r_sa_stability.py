# -*- coding: utf-8 -*-
"""
lesson3_2_estimate_r_sa_stability.py

整体在干什么？
1) 复用一个极简 MDP（TinyMDP），在固定(s,a)下产生奖励样本 r_i。
2) 用样本均值估计 r(s,a) = E[R_{t+1} | s,a]，并估计样本方差与标准误差。
3) 对不同样本量 N（例如 10/50/200/1000），重复做 many trials（例如 200 次）：
   - 统计平均绝对误差 |r_hat - r_true|
   - 统计平均置信区间半宽（half-width）
   - 统计“真值是否落在95%置信区间内”的覆盖率
4) 直观看到：误差与区间宽度大致按 1/sqrt(N) 缩小。

你需要从这份代码掌握的概念：
- r_hat 用样本均值估计
- 误差来源：奖励噪声导致方差；样本量 N 决定稳定性
- 置信区间只是近似（CLT/正态近似），但足够用于工程直觉
"""

import math
import random

A0, A1 = 0, 1


class TinyMDP:
    """
    极简 MDP（用于固定(s,a)下的奖励采样）：
    - 状态：0 或 1
    - 动作：0 或 1
    - 转移：P(s'=1 | s,a) 已知（我们用它算真值 r(s,a)）
    - 奖励：base_reward(s,a,s') + 高斯噪声（均值0，方差由 noise_std 控制）
    """

    def __init__(self, noise_std: float = 0.3, seed: int = 0):
        self.noise_std = noise_std
        self.rng = random.Random(seed)

        self.p_to_1 = {
            (0, A0): 0.7,
            (0, A1): 0.4,
            (1, A0): 1.0,
            (1, A1): 1.0,
        }

    def base_reward(self, s: int, a: int, s2: int) -> float:
        """
        确定性奖励规则（不含噪声）：
        - 到达 s'=1 奖励 +1，否则 0
        - a=1 有固定成本 0.1
        """
        reach_bonus = 1.0 if s2 == 1 else 0.0
        action_cost = 0.1 if a == A1 else 0.0
        return reach_bonus - action_cost

    def step(self, s: int, a: int):
        """
        采样一步交互，返回 (s2, r_sample)：
        1) 按 P(s'|s,a) 采样 s2
        2) 奖励样本 r = base_reward + N(0, noise_std^2)
        """
        p = self.p_to_1[(s, a)]
        s2 = 1 if self.rng.random() < p else 0
        noise = self.rng.gauss(0.0, self.noise_std)
        r = self.base_reward(s, a, s2) + noise
        return s2, r

    def true_r_s_a(self, s: int, a: int) -> float:
        """
        理论真值：r(s,a) = E[R_{t+1} | s,a]
        噪声均值为0，因此只需对 s' 的分布做加权平均。
        """
        p1 = self.p_to_1[(s, a)]
        p0 = 1.0 - p1
        return p0 * self.base_reward(s, a, 0) + p1 * self.base_reward(s, a, 1)


def sample_rewards(env: TinyMDP, s: int, a: int, n: int):
    """
    固定(s,a)，采样 n 个奖励样本。
    返回 rewards 列表。
    """
    rewards = []
    for _ in range(n):
        _, r = env.step(s, a)
        rewards.append(r)
    return rewards


def mean_and_unbiased_var(xs):
    """
    计算样本均值与无偏样本方差（ddof=1）。
    若样本量<2，方差返回 0.0（此时置信区间没有意义，只做演示）。
    """
    n = len(xs)
    mu = sum(xs) / n
    if n < 2:
        return mu, 0.0
    sse = sum((x - mu) ** 2 for x in xs)
    var = sse / (n - 1)
    return mu, var


def approx_95ci_halfwidth(sample_var: float, n: int) -> float:
    """
    用正态近似（CLT）构造 95% 置信区间半宽：
    halfwidth = 1.96 * sqrt(var / n)

    注意：这是近似，用于工程直觉；严格小样本应使用 t 分布。
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if sample_var <= 0.0:
        return 0.0
    return 1.96 * math.sqrt(sample_var / n)


def run_trials(noise_std: float, s: int, a: int, n_list, trials: int = 200, seed: int = 0):
    """
    在给定噪声强度下，对不同样本量 N 做重复试验：
    - 每次试验：采样 N 个奖励，得到 r_hat、样本方差、CI 半宽
    - 汇总：平均绝对误差、平均CI半宽、覆盖率（真值是否落在CI内）

    返回：dict[N] = (avg_abs_err, avg_halfwidth, coverage)
    """
    # 为了让不同 N 的 trials 可复现，同时又不完全相同，使用不同子种子
    base_rng = random.Random(seed)

    results = {}
    for n in n_list:
        abs_err_sum = 0.0
        half_sum = 0.0
        covered = 0

        for _ in range(trials):
            env = TinyMDP(noise_std=noise_std, seed=base_rng.randrange(10**9))
            r_true = env.true_r_s_a(s, a)

            rewards = sample_rewards(env, s, a, n)
            r_hat, var_hat = mean_and_unbiased_var(rewards)
            half = approx_95ci_halfwidth(var_hat, n)

            abs_err_sum += abs(r_hat - r_true)
            half_sum += half
            if (r_true >= r_hat - half) and (r_true <= r_hat + half):
                covered += 1

        results[n] = (
            abs_err_sum / trials,
            half_sum / trials,
            covered / trials
        )

    return results


if __name__ == "__main__":
    s, a = 0, A0
    n_list = [10, 50, 200, 1000]
    trials = 300

    for noise_std in [0.1, 0.3, 0.8]:
        print(f"\n=== Noise std = {noise_std} | fixed (s={s}, a={a}) ===")
        res = run_trials(noise_std=noise_std, s=s, a=a, n_list=n_list, trials=trials, seed=42)

        print("N    | avg |r_hat - r_true|   avg CI halfwidth   coverage (approx 95%)")
        print("-----+---------------------+--------------------+------------------------")
        for n in n_list:
            avg_abs_err, avg_half, cov = res[n]
            print(f"{n:>4} | {avg_abs_err:>19.4f} | {avg_half:>18.4f} | {cov:>22.3f}")
