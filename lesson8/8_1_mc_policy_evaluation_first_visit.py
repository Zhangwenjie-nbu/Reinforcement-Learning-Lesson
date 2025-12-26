# -*- coding: utf-8 -*-
"""
lesson8_1_mc_policy_evaluation_first_visit.py

整体在干什么？
1) 构造一个“黑箱”环境：1D Random Walk（可带 slip），我们只通过 step() 交互获取样本，不使用显式转移概率求和。
2) 固定一个策略 π（均匀随机策略），用于生成 episode（on-policy）。
3) 采样很多条 episode。对每条 episode：
   - 记录状态序列 S0,S1,... 以及奖励序列 R1,R2,...
   - 计算每个时间步 t 的回报 G_t
   - 对每个状态 s，只在该 episode 的“首次访问”时，用对应 G_t 作为样本更新 V(s)
4) 用“累计均值”维护 V(s) 的估计值：
      V_hat(s) = average of observed returns from first-visits to s

你需要掌握：
- MC 不用模型，不用贝尔曼方程；它直接用回报样本均值估计期望
- First-Visit：每个 episode 对某个状态只更新一次
- V_hat 的收敛需要足够多 episode，且状态要被访问到足够多次
"""

import random
from collections import defaultdict

LEFT, RIGHT = 0, 1


class SlipperyRandomWalk:
    """
    1D Random Walk（可打滑）：
    - 状态：0..(n_states-1)
    - 起点：start_state
    - 终止：0 与 n_states-1
    - 动作：LEFT/RIGHT
    - slip：以 slip_prob 概率动作反转
    - 奖励：到达右端终止态给 +1，否则 0
    """

    def __init__(self, n_states=7, start_state=3, slip_prob=0.2, seed=0):
        self.n_states = n_states
        self.start_state = start_state
        self.slip_prob = slip_prob
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
        判断是否为终止态。
        """
        return s == self.terminal_left or s == self.terminal_right

    def step(self, s: int, a: int):
        """
        与环境交互一步，返回 (s2, r, done)。
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
        r = 1.0 if s2 == self.terminal_right else 0.0
        return s2, r, done


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


def generate_episode(env: SlipperyRandomWalk, policy: UniformRandomPolicy, max_steps=200):
    """
    采样一条 episode：
    返回：
    - states: [S0, S1, ..., ST]
    - rewards: [R1, R2, ..., RT]  与 states 对齐（reward_t 是从 S_{t-1} 到 S_t 的奖励）
    """
    s = env.reset()
    states = [s]
    rewards = []

    for _ in range(max_steps):
        a = policy.act(s)
        s, r, done = env.step(s, a)
        states.append(s)
        rewards.append(r)
        if done:
            break

    return states, rewards


def compute_returns(rewards, gamma: float):
    """
    给定 rewards=[R1,...,RT]，计算每个时间步的回报 G_t（t从0到T-1，对应状态 S_t）。
    返回 returns=[G0,G1,...,G_{T-1}]
    注意：
    - rewards 的长度为 T
    - states 的长度为 T+1
    - G_t = sum_{k=t}^{T-1} gamma^{k-t} * R_{k+1}
    """
    T = len(rewards)
    returns = [0.0 for _ in range(T)]
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


def mc_policy_evaluation_first_visit(env, policy, gamma: float, n_episodes: int):
    """
    First-Visit Monte Carlo 策略评估：
    - 对每个状态维护：
      - sum_returns[s]：累计回报和
      - count[s]：首次访问计数
      - V_hat[s]：均值估计

    返回：
    - V_hat: dict[s] -> estimated V^π(s)
    - count: dict[s] -> how many first-visit samples collected
    """
    sum_returns = defaultdict(float)
    count = defaultdict(int)
    V_hat = defaultdict(float)

    for ep in range(1, n_episodes + 1):
        states, rewards = generate_episode(env, policy)
        returns = compute_returns(rewards, gamma)

        visited_in_episode = set()

        # 对 episode 中每个时间步 t（对应状态 S_t），做 First-Visit 更新
        for t, s in enumerate(states[:-1]):  # 最后一个状态是终止态，不对应 returns
            if s in visited_in_episode:
                continue
            visited_in_episode.add(s)

            G_t = returns[t]
            sum_returns[s] += G_t
            count[s] += 1
            V_hat[s] = sum_returns[s] / count[s]

        # 适度打印学习进度（避免过多输出）
        if ep in [10, 50, 200, 1000, n_episodes]:
            # 只展示常见的非终止态估计
            keys = sorted([k for k in V_hat.keys() if k not in (env.terminal_left, env.terminal_right)])
            snapshot = {k: round(V_hat[k], 4) for k in keys}
            print(f"episode={ep:>5} | V_hat(non-terminal)={snapshot}")

    return V_hat, count


if __name__ == "__main__":
    gamma = 0.95
    n_episodes = 5000

    env = SlipperyRandomWalk(n_states=7, start_state=3, slip_prob=0.2, seed=42)
    policy = UniformRandomPolicy(seed=7)

    V_hat, count = mc_policy_evaluation_first_visit(env, policy, gamma, n_episodes)

    print("\nFinal estimates (state: V_hat, count):")
    for s in range(env.n_states):
        print(f"s={s}: V_hat={V_hat[s]:.6f} | count={count[s]}")
