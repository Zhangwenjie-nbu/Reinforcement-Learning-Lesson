# -*- coding: utf-8 -*-
"""
lesson5_1_policy_objects.py

整体在干什么？
1) 用一个简单的离散环境（1D Random Walk）提供状态与动作空间背景。
2) 实现三种策略（policy），展示“策略是从状态到动作分布的映射”：
   - UniformRandomPolicy：π(a|s) 固定为均匀分布
   - EpsilonGreedyPolicy：基于 Q(s,a) 的 ε-greedy 随机策略（把探索写进分布）
   - DeterministicRightPolicy：确定性策略 a=μ(s)，永远选 RIGHT
3) 在固定状态 s=3 下，多次从策略采样动作，统计频率，观察：
   - 随机策略的频率接近其概率分布
   - 确定性策略的频率集中在一个动作上
4) 这为后续“价值函数 V^π / Q^π 的定义”做准备：价值是“按策略采样行为”下的期望回报。

你需要掌握：
- π(a|s) 是分布；act(s) 是从该分布采样的过程
- ε-greedy 是随机策略，不是“一个动作”
- 确定性策略可以视为随机策略的退化情形
"""

import random
from collections import Counter

LEFT, RIGHT = 0, 1


class RandomWalkEnv:
    """
    1D Random Walk 环境（仅用于提供状态/动作背景，本节不关注奖励学习）：
    - 状态：0..6
    - 起点：3
    - 终止：0 与 6
    - 动作：LEFT/RIGHT
    """

    def __init__(self, n_states=7, start_state=3):
        self.n_states = n_states
        self.start_state = start_state
        self.terminal_left = 0
        self.terminal_right = n_states - 1

    def reset(self):
        """
        重置到起点，返回初始状态。
        """
        return self.start_state


class UniformRandomPolicy:
    """
    随机策略：对任意状态 s，π(LEFT|s)=π(RIGHT|s)=0.5
    """

    def __init__(self, rng=None):
        self.rng = rng or random.Random(0)

    def act(self, s: int) -> int:
        """
        从均匀分布采样动作：A ~ Uniform({LEFT, RIGHT})
        """
        return self.rng.choice([LEFT, RIGHT])

    def prob(self, s: int, a: int) -> float:
        """
        返回 π(a|s) 的概率值（本策略与 s 无关）。
        """
        return 0.5


class DeterministicRightPolicy:
    """
    确定性策略：对任意状态 s，总是选择 RIGHT。
    可视为随机策略的特例：π(RIGHT|s)=1, π(LEFT|s)=0
    """

    def act(self, s: int) -> int:
        """
        确定性输出动作：a = RIGHT
        """
        return RIGHT

    def prob(self, s: int, a: int) -> float:
        """
        返回退化分布的概率质量。
        """
        return 1.0 if a == RIGHT else 0.0


class EpsilonGreedyPolicy:
    """
    ε-greedy 随机策略（离散动作）：
    - 以 (1-ε) 概率选择 argmax_a Q(s,a)
    - 以 ε 概率在所有动作中均匀随机选一个（探索）

    注意：这是一个随机策略，π(a|s) 可显式写出。
    """

    def __init__(self, Q, epsilon: float = 0.1, rng=None):
        self.Q = Q  # dict[(s,a)] -> value
        self.epsilon = epsilon
        self.rng = rng or random.Random(0)

    def greedy_action(self, s: int) -> int:
        """
        计算贪心动作 argmax_a Q(s,a)。
        若并列最大，则随机选一个，避免固定偏置。
        """
        q_left = self.Q[(s, LEFT)]
        q_right = self.Q[(s, RIGHT)]
        max_q = max(q_left, q_right)
        candidates = []
        if q_left == max_q:
            candidates.append(LEFT)
        if q_right == max_q:
            candidates.append(RIGHT)
        return self.rng.choice(candidates)

    def act(self, s: int) -> int:
        """
        从 ε-greedy 分布采样动作。
        """
        if self.rng.random() < self.epsilon:
            return self.rng.choice([LEFT, RIGHT])
        return self.greedy_action(s)

    def prob(self, s: int, a: int) -> float:
        """
        显式给出 π(a|s)：
        - 探索部分：ε * 1/|A|
        - 利用部分： (1-ε) * I[a 是贪心动作]（若并列，分摊）
        """
        # 探索概率
        p = self.epsilon * 0.5

        q_left = self.Q[(s, LEFT)]
        q_right = self.Q[(s, RIGHT)]
        max_q = max(q_left, q_right)

        greedy_actions = []
        if q_left == max_q:
            greedy_actions.append(LEFT)
        if q_right == max_q:
            greedy_actions.append(RIGHT)

        if a in greedy_actions:
            p += (1.0 - self.epsilon) / len(greedy_actions)
        return p


def sample_action_frequencies(policy, s: int, n: int = 20000):
    """
    在固定状态 s 下，从策略采样 n 次动作，统计频率（近似策略分布）。
    """
    cnt = Counter()
    for _ in range(n):
        a = policy.act(s)
        cnt[a] += 1

    freqs = {
        LEFT: cnt[LEFT] / n,
        RIGHT: cnt[RIGHT] / n
    }
    return freqs


if __name__ == "__main__":
    env = RandomWalkEnv()
    s = env.reset()

    # 构造一个简单 Q(s,a)（演示用途）：
    # 设想 RIGHT 更优：Q(s,RIGHT)=1, Q(s,LEFT)=0
    Q = {(state, LEFT): 0.0 for state in range(env.n_states)}
    Q.update({(state, RIGHT): 1.0 for state in range(env.n_states)})

    policies = [
        ("UniformRandom", UniformRandomPolicy(rng=random.Random(1))),
        ("EpsilonGreedy(eps=0.1)", EpsilonGreedyPolicy(Q, epsilon=0.1, rng=random.Random(2))),
        ("DeterministicRight", DeterministicRightPolicy()),
    ]

    print(f"Fixed state s = {s}\n")
    for name, pi in policies:
        freqs = sample_action_frequencies(pi, s, n=50000)

        # 如果策略有 prob 方法，则打印其理论分布
        p_left = pi.prob(s, LEFT) if hasattr(pi, "prob") else None
        p_right = pi.prob(s, RIGHT) if hasattr(pi, "prob") else None

        print(f"=== Policy: {name} ===")
        print(f"Empirical freq: P(LEFT)≈{freqs[LEFT]:.3f}, P(RIGHT)≈{freqs[RIGHT]:.3f}")
        if p_left is not None:
            print(f"Theoretical π: P(LEFT)={p_left:.3f}, P(RIGHT)={p_right:.3f}")
        print()
