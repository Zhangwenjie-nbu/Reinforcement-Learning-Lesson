# -*- coding: utf-8 -*-
"""
lesson5_3_q_value_and_relation.py

整体在干什么？
1) 在一个带随机性的 1D Random Walk 环境中（含 slip 打滑），固定一个策略 π（均匀随机）。
2) 通过采样估计：
   - V^π(s0)：从 s0 出发按 π 行为的回报期望
   - Q^π(s0,a)：从 s0 出发第一步强制执行动作 a，之后按 π 行为的回报期望
3) 验证定义层面的关系式：
      V^π(s0) = Σ_a π(a|s0) Q^π(s0,a)
   对均匀随机策略来说：V ≈ 0.5*Q(s0,LEFT) + 0.5*Q(s0,RIGHT)

你需要掌握：
- Q^π(s,a) 比 V^π(s) 多“固定第一步动作”
- V 是对 Q 按策略概率加权求和（全概率/条件期望分解）
- 这些都在定义层面成立，不依赖贝尔曼方程
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
    - 转移：以 slip_prob 概率动作反转
    - 奖励：到达右端终止态给 +1，否则 0
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
        1) slip：以 slip_prob 概率动作反转
        2) 执行动作得到下一状态
        3) 给出奖励与终止标记
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
    均匀随机策略：π(LEFT|s)=π(RIGHT|s)=0.5
    """

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, s: int) -> int:
        """
        从均匀分布采样动作。
        """
        return self.rng.choice([LEFT, RIGHT])

    def prob(self, s: int, a: int) -> float:
        """
        返回 π(a|s)。
        """
        return 0.5


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


def run_episode_from_state(env: SlipperyRandomWalk, policy: UniformRandomPolicy, gamma: float, s0: int, max_steps=200):
    """
    从指定状态 s0 开始按策略采样一条 episode，返回折扣回报 G0。

    注意：
    - 这里不调用 env.reset()，而是用传入的 s0，便于估计 V^π(s0) 与 Q^π(s0,a)。
    """
    s = s0
    rewards = []

    for _ in range(max_steps):
        a = policy.act(s)
        s, r, done = env.step(s, a)
        rewards.append(r)
        if done:
            break

    return discounted_return(rewards, gamma)


def run_episode_with_forced_first_action(env: SlipperyRandomWalk, policy: UniformRandomPolicy, gamma: float, s0: int, a0: int, max_steps=200):
    """
    从状态 s0 开始采样一条 episode，但第一步动作强制为 a0：
    - 第一步用 a0 与环境交互
    - 从第二步开始按策略 π 采样动作
    - 返回折扣回报 G0

    这正对应 Q^π(s0,a0) 的定义语义。
    """
    s = s0
    rewards = []

    # forced first step
    s, r, done = env.step(s, a0)
    rewards.append(r)
    if done:
        return discounted_return(rewards, gamma)

    # follow policy afterwards
    for _ in range(max_steps - 1):
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


def estimate_V(env_seed: int, policy_seed: int, s0: int, gamma: float, slip_prob: float, n_episodes: int):
    """
    通过采样直接估计 V^π(s0)：
    - 每条 episode 从 s0 开始按 π 走
    - 返回均值、方差、CI 半宽
    """
    env = SlipperyRandomWalk(slip_prob=slip_prob, seed=env_seed)
    policy = UniformRandomPolicy(seed=policy_seed)

    returns = [run_episode_from_state(env, policy, gamma, s0) for _ in range(n_episodes)]
    mu, var = mean_and_unbiased_var(returns)
    half = approx_95ci_halfwidth(var, n_episodes)
    return mu, var, half


def estimate_Q(env_seed: int, policy_seed: int, s0: int, a0: int, gamma: float, slip_prob: float, n_episodes: int):
    """
    通过采样估计 Q^π(s0,a0)：
    - 每条 episode 第一动作强制为 a0
    - 之后按 π 行为
    - 返回均值、方差、CI 半宽
    """
    env = SlipperyRandomWalk(slip_prob=slip_prob, seed=env_seed)
    policy = UniformRandomPolicy(seed=policy_seed)

    returns = [run_episode_with_forced_first_action(env, policy, gamma, s0, a0) for _ in range(n_episodes)]
    mu, var = mean_and_unbiased_var(returns)
    half = approx_95ci_halfwidth(var, n_episodes)
    return mu, var, half


if __name__ == "__main__":
    gamma = 0.95
    slip_prob = 0.2
    s0 = 3
    n = 3000

    # 直接估计 V^π(s0)
    V_hat, V_var, V_half = estimate_V(env_seed=42, policy_seed=7, s0=s0, gamma=gamma, slip_prob=slip_prob, n_episodes=n)

    # 分别估计 Q^π(s0,LEFT) 与 Q^π(s0,RIGHT)
    QL_hat, QL_var, QL_half = estimate_Q(env_seed=43, policy_seed=7, s0=s0, a0=LEFT, gamma=gamma, slip_prob=slip_prob, n_episodes=n)
    QR_hat, QR_var, QR_half = estimate_Q(env_seed=44, policy_seed=7, s0=s0, a0=RIGHT, gamma=gamma, slip_prob=slip_prob, n_episodes=n)

    # 用关系式间接计算 V：V = Σ_a π(a|s0) Q(s0,a)
    # 这里 π 是均匀随机，所以权重均为0.5
    V_from_Q = 0.5 * QL_hat + 0.5 * QR_hat

    print(f"Fixed s0={s0}, gamma={gamma}, slip_prob={slip_prob}, episodes={n}\n")

    print("=== Direct estimate of V^π(s0) ===")
    print(f"V_hat = {V_hat:.4f} ± {V_half:.4f} (95% CI half)\n")

    print("=== Estimates of Q^π(s0,a) ===")
    print(f"Q_hat(s0,LEFT)  = {QL_hat:.4f} ± {QL_half:.4f}")
    print(f"Q_hat(s0,RIGHT) = {QR_hat:.4f} ± {QR_half:.4f}\n")

    print("=== Check relation: V^π(s0) ?= Σ_a π(a|s0) Q^π(s0,a) ===")
    print(f"V_from_Q = 0.5*Q(LEFT) + 0.5*Q(RIGHT) = {V_from_Q:.4f}")
    print(f"Difference |V_hat - V_from_Q| = {abs(V_hat - V_from_Q):.4f}")
