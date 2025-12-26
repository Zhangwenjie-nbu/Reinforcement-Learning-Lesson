# -*- coding: utf-8 -*-
"""
lesson8_3_td0_bellman_consistency_monitor.py

整体在干什么？
1) 构造一个黑箱环境：1D Random Walk（可打滑），固定均匀随机策略 π。
2) 用 TD(0) 做策略评估更新 V(s)。
3) 额外监控“贝尔曼一致性误差”的样本版本：TD 误差 δ = r + gamma*V(s') - V(s)。
4) 对每个状态 s 维护：
   - delta_sum[s]：累计 δ
   - delta_count[s]：δ 样本数
   - mean_delta[s]：平均 δ
   观察训练过程中 mean_delta[s] 是否趋近 0。

你需要掌握：
- TD(0) 在期望意义下逼近贝尔曼期望算子固定点
- 当接近固定点时，TD 误差在条件期望上应接近 0
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
        return self.start_state

    def is_terminal(self, s: int) -> bool:
        return s == self.terminal_left or s == self.terminal_right

    def step(self, s: int, a: int):
        if self.is_terminal(s):
            return s, 0.0, True

        if self.rng.random() < self.slip_prob:
            a = LEFT if a == RIGHT else RIGHT

        s2 = max(self.terminal_left, s - 1) if a == LEFT else min(self.terminal_right, s + 1)
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
        return self.rng.choice([LEFT, RIGHT])


def td0_with_delta_monitor(env, policy, gamma: float, alpha: float, n_episodes: int, max_steps=200):
    """
    TD(0) 策略评估 + TD误差均值监控：
    - 每步更新 V(s)
    - 同时累计 δ 的均值，用来观察“贝尔曼一致性误差”是否趋近0

    返回：
    - V: dict[int,float]
    - mean_delta: dict[int,float]
    - delta_count: dict[int,int]
    """
    V = defaultdict(float)
    delta_sum = defaultdict(float)
    delta_count = defaultdict(int)

    # 终止态固定为0
    V[env.terminal_left] = 0.0
    V[env.terminal_right] = 0.0

    checkpoints = [10, 50, 200, 1000, n_episodes]

    for ep in range(1, n_episodes + 1):
        s = env.reset()

        for _ in range(max_steps):
            a = policy.act(s)
            s2, r, done = env.step(s, a)

            if not env.is_terminal(s):
                v_next = 0.0 if env.is_terminal(s2) else V[s2]
                td_target = r + gamma * v_next
                delta = td_target - V[s]

                # 监控 delta（用更新前的 V 计算的 δ，更符合“误差”语义）
                delta_sum[s] += delta
                delta_count[s] += 1

                # TD(0) 更新
                V[s] += alpha * delta

            s = s2
            if done:
                break

        if ep in checkpoints:
            mean_delta = {s: (delta_sum[s] / delta_count[s]) if delta_count[s] > 0 else 0.0
                          for s in range(env.n_states)}
            snapshot = {s: round(mean_delta[s], 4) for s in range(env.n_states)}
            print(f"episode={ep:>5} | mean_delta={snapshot}")

    mean_delta = {s: (delta_sum[s] / delta_count[s]) if delta_count[s] > 0 else 0.0
                  for s in range(env.n_states)}
    return V, mean_delta, delta_count


if __name__ == "__main__":
    gamma = 0.95
    alpha = 0.1
    n_episodes = 5000

    env = SlipperyRandomWalk(n_states=7, start_state=3, slip_prob=0.2, seed=42)
    policy = UniformRandomPolicy(seed=7)

    V, mean_delta, delta_count = td0_with_delta_monitor(env, policy, gamma, alpha, n_episodes)

    print("\nFinal V_hat:")
    for s in range(env.n_states):
        print(f"s={s}: V_hat={V[s]:.6f}")

    print("\nFinal mean TD error (mean_delta) and counts:")
    for s in range(env.n_states):
        print(f"s={s}: mean_delta={mean_delta[s]:.6f} | count={delta_count[s]}")
