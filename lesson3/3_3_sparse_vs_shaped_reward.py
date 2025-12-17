# -*- coding: utf-8 -*-
"""
lesson3_3_sparse_vs_shaped_reward.py

整体在干什么？
1) 构造一个最小 1D Random Walk 环境（状态0..6，起点3，终止在0或6）。
2) 对比两种奖励设计：
   - sparse：仅到达右端终止状态6时奖励=+1，其余为0（典型稀疏奖励）
   - shaped：在 sparse 的基础上加入 potential-based shaping：
            r_shaped = r_sparse + gamma*phi(s') - phi(s)
            其中 phi(s)=s/(N-1)，提供“朝右走更好”的密集学习信号
3) 在随机策略下采样大量 episode，统计：
   - 非零奖励比例（每一步是否出现非零奖励）
   - episode 回报均值的估计误差/置信区间随样本量变化
4) 直观展示：奖励越稀疏，有效学习信号越少，估计越不稳定，越“难学”。

你需要掌握的概念：
- 稀疏奖励：大部分时间步 reward=0，信号密度低
- 塑形奖励：提高信号密度，通常能降低方差、提升样本效率
- potential-based shaping：一种较“安全”的塑形方式（常用于不改变最优策略）
"""

import math
import random

LEFT, RIGHT = 0, 1


class RandomWalkEnv:
    """
    1D Random Walk 环境：
    - 状态：0..(n_states-1)
    - 起点：start_state
    - 终止：0 与 n_states-1
    - 动作：LEFT/RIGHT，确定性转移（边界夹紧到终止态）
    - 奖励：由 reward_mode 决定（sparse 或 shaped）
    """

    def __init__(self, n_states=7, start_state=3, gamma=0.95, reward_mode="sparse", seed=0):
        self.n_states = n_states
        self.start_state = start_state
        self.gamma = gamma
        self.reward_mode = reward_mode
        self.rng = random.Random(seed)

        self.terminal_left = 0
        self.terminal_right = n_states - 1

    def reset(self):
        """
        重置环境到起点，返回初始状态。
        """
        return self.start_state

    def is_terminal(self, s: int) -> bool:
        """
        判断是否为终止状态。
        """
        return s == self.terminal_left or s == self.terminal_right

    def phi(self, s: int) -> float:
        """
        势函数 phi(s)，用于 potential-based shaping。
        这里用一个简单线性势：phi(s)=s/(N-1)，越靠右势越高。
        """
        return s / (self.n_states - 1)

    def step(self, s: int, a: int):
        """
        环境一步交互：
        1) 根据动作得到下一状态 s2
        2) 根据 reward_mode 生成即时奖励 r
        3) 返回 (s2, r, done)

        注意：
        - sparse：只有到达右端终止态时 r=+1，否则 0
        - shaped：r = r_sparse + gamma*phi(s2) - phi(s)
        """
        if self.is_terminal(s):
            return s, 0.0, True

        if a == LEFT:
            s2 = max(self.terminal_left, s - 1)
        else:
            s2 = min(self.terminal_right, s + 1)

        done = self.is_terminal(s2)

        # sparse reward rule
        r_sparse = 1.0 if s2 == self.terminal_right else 0.0

        if self.reward_mode == "sparse":
            r = r_sparse
        elif self.reward_mode == "shaped":
            r = r_sparse + self.gamma * self.phi(s2) - self.phi(s)
        else:
            raise ValueError("reward_mode must be 'sparse' or 'shaped'.")

        return s2, r, done

    def random_policy(self, s: int) -> int:
        """
        一个简单随机策略：以 0.5/0.5 选择 LEFT 或 RIGHT。
        用它来展示“早期/无探索偏置策略”下稀疏奖励的困难。
        """
        return self.rng.choice([LEFT, RIGHT])


def run_episode(env: RandomWalkEnv, max_steps=100):
    """
    采样一条 episode：
    - 逐步用 env.random_policy 选择动作
    - 收集 rewards
    - 返回 rewards 列表与 step 数
    """
    s = env.reset()
    rewards = []
    for _ in range(max_steps):
        a = env.random_policy(s)
        s, r, done = env.step(s, a)
        rewards.append(r)
        if done:
            break
    return rewards, len(rewards)


def discounted_return(rewards, gamma: float) -> float:
    """
    计算折扣回报 G0 = r1 + gamma*r2 + ...
    """
    G = 0.0
    power = 1.0
    for r in rewards:
        G += power * r
        power *= gamma
    return G


def mean_and_unbiased_var(xs):
    """
    计算样本均值与无偏样本方差（ddof=1）。
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
    用正态近似构造 95% CI 半宽：1.96*sqrt(var/n)
    """
    if n < 2 or sample_var <= 0.0:
        return 0.0
    return 1.96 * math.sqrt(sample_var / n)


def evaluate(env: RandomWalkEnv, n_episodes: int, seed: int = 0):
    """
    在给定环境与奖励模式下，采样 n_episodes 条 episode，并统计：
    - 回报均值、方差、95%CI半宽
    - 非零奖励比例（reward density）
    - 平均 episode 长度
    """
    # 重新设定随机种子，保证可复现
    env.rng = random.Random(seed)

    returns = []
    nonzero_rewards = 0
    total_rewards_count = 0
    lengths = []

    for _ in range(n_episodes):
        rewards, L = run_episode(env)
        lengths.append(L)

        total_rewards_count += len(rewards)
        nonzero_rewards += sum(1 for r in rewards if abs(r) > 1e-12)

        returns.append(discounted_return(rewards, env.gamma))

    mu, var = mean_and_unbiased_var(returns)
    half = approx_95ci_halfwidth(var, len(returns))

    density = nonzero_rewards / max(1, total_rewards_count)
    avg_len = sum(lengths) / len(lengths)

    return {
        "mean_return": mu,
        "var_return": var,
        "ci_halfwidth": half,
        "reward_density": density,
        "avg_ep_len": avg_len,
    }


if __name__ == "__main__":
    gamma = 0.95
    n_states = 7
    start_state = 3

    # 对比不同 episode 数量下的估计稳定性
    episode_counts = [50, 200, 1000, 5000]

    for mode in ["sparse", "shaped"]:
        print(f"\n=== Reward mode: {mode} ===")
        for n_ep in episode_counts:
            env = RandomWalkEnv(
                n_states=n_states,
                start_state=start_state,
                gamma=gamma,
                reward_mode=mode,
                seed=0
            )
            stats = evaluate(env, n_episodes=n_ep, seed=42)

            print(
                f"episodes={n_ep:>5} | "
                f"mean_return={stats['mean_return']:.4f} ± {stats['ci_halfwidth']:.4f} (95% CI half) | "
                f"reward_density={stats['reward_density']:.3f} | "
                f"avg_ep_len={stats['avg_ep_len']:.2f}"
            )
