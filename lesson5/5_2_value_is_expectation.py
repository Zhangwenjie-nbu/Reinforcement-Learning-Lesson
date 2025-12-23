# -*- coding: utf-8 -*-
"""
lesson5_2_value_is_expectation.py

整体在干什么？
1) 构造一个带随机性的 MDP（1D Random Walk + slip），确保从同一状态出发也会产生不同轨迹与不同回报。
2) 固定一个策略 π（这里用均匀随机策略），强调“策略已固定，唯一变化来自随机性”。
3) 从同一初始状态 s0 多次采样 episode，计算每条 episode 的折扣回报 G0。
4) 用样本均值估计 V^π(s0)：
      V^π(s0) = E[G0 | S0=s0]
   并展示估计的波动与置信区间随样本量增加而收缩。

你需要掌握：
- 一条 episode 给你的是 G0 的“样本值”，不是 V^π(s0)
- V^π(s0) 是对策略随机性与环境随机性取期望得到的量
- 只能通过多次采样用统计方式估计
"""

import math
import random

LEFT, RIGHT = 0, 1


class SlipperyRandomWalk:
    """
    带打滑随机性的 1D Random Walk：
    - 状态：0..(n_states-1)
    - 起点：start_state
    - 终止：0 与 n_states-1
    - 动作：LEFT/RIGHT
    - 转移：以 slip_prob 概率动作反转（增加环境随机性）
    - 奖励：到达右端终止态给 +1，否则 0（稀疏奖励）
    """

    def __init__(self, n_states=7, start_state=3, slip_prob=0.2, seed=0):
        self.n_states = n_states
        self.start_state = start_state
        self.slip_prob = slip_prob
        self.rng = random.Random(seed)

        self.terminal_left = 0
        self.terminal_right = n_states - 1

    def reset(self) -> int:
        """
        重置到起点，返回初始状态。
        """
        return self.start_state

    def is_terminal(self, s: int) -> bool:
        """
        判断是否为终止状态。
        """
        return s == self.terminal_left or s == self.terminal_right

    def step(self, s: int, a: int):
        """
        环境一步交互：
        1) 以 slip_prob 概率将动作反转（增加随机性）
        2) 执行动作得到下一状态（边界夹紧到终止态）
        3) 返回 (s2, reward, done)

        奖励规则：
        - 到达右端终止态：reward=+1
        - 否则：reward=0
        """
        if self.is_terminal(s):
            return s, 0.0, True

        # slip: flip action
        if self.rng.random() < self.slip_prob:
            a = LEFT if a == RIGHT else RIGHT

        if a == LEFT:
            s2 = max(self.terminal_left, s - 1)
        else:
            s2 = min(self.terminal_right, s + 1)

        done = self.is_terminal(s2)
        reward = 1.0 if s2 == self.terminal_right else 0.0
        return s2, reward, done


class UniformRandomPolicy:
    """
    均匀随机策略：对任意状态 s，π(LEFT|s)=π(RIGHT|s)=0.5
    """

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, s: int) -> int:
        """
        从均匀分布采样动作。
        """
        return self.rng.choice([LEFT, RIGHT])


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


def run_episode(env: SlipperyRandomWalk, policy: UniformRandomPolicy, gamma: float, max_steps=200):
    """
    采样一条 episode：
    - 从 env.reset() 开始
    - 每步用 policy.act(s) 选动作
    - 与环境交互得到奖励序列
    - 返回该 episode 的折扣回报 G0
    """
    s = env.reset()
    rewards = []

    for _ in range(max_steps):
        a = policy.act(s)
        s, r, done = env.step(s, a)
        rewards.append(r)
        if done:
            break

    return discounted_return(rewards, gamma)


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


def estimate_value(env_seed: int, policy_seed: int, n_episodes: int, gamma: float, slip_prob: float):
    """
    估计 V^π(start_state)：
    - 用固定策略 π
    - 采样 n_episodes 条 episode
    - 返回估计均值、方差与 95%CI 半宽
    """
    env = SlipperyRandomWalk(slip_prob=slip_prob, seed=env_seed)
    policy = UniformRandomPolicy(seed=policy_seed)

    returns = [run_episode(env, policy, gamma) for _ in range(n_episodes)]
    mu, var = mean_and_unbiased_var(returns)
    half = approx_95ci_halfwidth(var, n_episodes)
    return mu, var, half


if __name__ == "__main__":
    gamma = 0.95
    slip_prob = 0.2

    # 不同采样规模下，观察 V^π 的估计如何收敛与区间如何缩小
    for n in [20, 100, 500, 2000]:
        mu, var, half = estimate_value(
            env_seed=42,
            policy_seed=7,
            n_episodes=n,
            gamma=gamma,
            slip_prob=slip_prob
        )
        print(f"episodes={n:>4} | V_hat(start)={mu:.4f} ± {half:.4f} (95% CI half) | var(G0)={var:.4f}")
