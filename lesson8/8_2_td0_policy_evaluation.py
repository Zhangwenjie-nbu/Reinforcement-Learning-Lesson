# -*- coding: utf-8 -*-
"""
lesson8_2_td0_policy_evaluation.py

整体在干什么？
1) 构造一个“黑箱”环境：1D Random Walk（可打滑），只能通过 reset/step 产生样本，不使用显式 P,R。
2) 固定一个策略 π（均匀随机），用于生成数据（on-policy）。
3) 实现 TD(0) 策略评估：
      V(s) <- V(s) + alpha * (r + gamma * V(s') - V(s))
   - 每一步交互都立刻更新（在线）
   - 终止态价值固定为0
4) 训练若干 episode 后打印 V 的估计，并展示学习进度。

你需要掌握：
- TD(0) 的目标是“一步 bootstrapping”：r + gamma*V(s')
- TD 误差 delta 衡量当前估计与一步一致性之间的差
- 相比 MC，它不必等 episode 结束，通常更高效，但目标含估计值会引入偏差
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


def td0_policy_evaluation(env, policy, gamma: float, alpha: float, n_episodes: int, max_steps=200):
    """
    TD(0) 策略评估：
    - V 初始化为0
    - 对每条 episode:
      从 reset() 开始按策略生成轨迹
      对每个转移 (s, r, s') 进行 TD(0) 更新：
        delta = r + gamma*V[s'] - V[s]
        V[s] += alpha*delta

    返回：
    - V: dict[int, float]
    - visit: dict[int, int] 记录各状态被更新次数（便于理解收敛质量）
    """
    V = defaultdict(float)
    visit = defaultdict(int)

    # 终止态固定为0（显式写出，避免被更新）
    V[env.terminal_left] = 0.0
    V[env.terminal_right] = 0.0

    for ep in range(1, n_episodes + 1):
        s = env.reset()

        for _ in range(max_steps):
            a = policy.act(s)
            s2, r, done = env.step(s, a)

            # TD(0) update（终止态不更新）
            if not env.is_terminal(s):
                td_target = r + gamma * (0.0 if env.is_terminal(s2) else V[s2])
                delta = td_target - V[s]
                V[s] += alpha * delta
                visit[s] += 1

            s = s2
            if done:
                break

        # 适度打印学习进度
        if ep in [10, 50, 200, 1000, n_episodes]:
            keys = list(range(env.n_states))
            snapshot = {k: round(V[k], 4) for k in keys}
            print(f"episode={ep:>5} | V_hat={snapshot}")

    return V, visit


if __name__ == "__main__":
    gamma = 0.95
    alpha = 0.1
    n_episodes = 5000

    env = SlipperyRandomWalk(n_states=7, start_state=3, slip_prob=0.2, seed=42)
    policy = UniformRandomPolicy(seed=7)

    V, visit = td0_policy_evaluation(env, policy, gamma, alpha, n_episodes)

    print("\nFinal estimates (state: V_hat, updates):")
    for s in range(env.n_states):
        print(f"s={s}: V_hat={V[s]:.6f} | updates={visit[s]}")
