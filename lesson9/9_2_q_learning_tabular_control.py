# -*- coding: utf-8 -*-
"""
lesson9_2_q_learning_tabular_control.py

整体在干什么？
1) 构造黑箱环境：1D Random Walk（可打滑），通过 reset/step 交互采样。
2) 实现 Q-learning（off-policy TD control）学习 Q(s,a)：
      Q(s,a) <- Q(s,a) + alpha * (r + gamma*max_{a'}Q(s',a') - Q(s,a))
3) 行为策略使用 ε-greedy 来探索产生数据（behavior policy）。
4) 更新目标使用 max（相当于目标策略是贪心策略），因此是 off-policy。
5) 训练过程中周期性评估：从起点用贪心策略跑多次，统计到达右端终止态比例。

你需要掌握：
- Q-learning 的关键在于 target 里用 max，而不是采样到的 a'
- 行为策略与目标策略不一致 => off-policy
- 在表格有限 MDP 中，合适条件下 Q-learning 可收敛到 Q*
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


def epsilon_greedy_action(Q, s: int, epsilon: float, rng: random.Random):
    """
    ε-greedy 行为策略：用于采样数据。
    """
    if rng.random() < epsilon:
        return rng.choice([LEFT, RIGHT])
    qL = Q[(s, LEFT)]
    qR = Q[(s, RIGHT)]
    return RIGHT if qR >= qL else LEFT


def greedy_action(Q, s: int):
    """
    纯贪心动作：用于评估成功率。
    """
    qL = Q[(s, LEFT)]
    qR = Q[(s, RIGHT)]
    return RIGHT if qR >= qL else LEFT


def evaluate_greedy_policy(env: SlipperyRandomWalk, Q, n_trials=200, max_steps=200):
    """
    从起点用贪心策略跑若干次，返回到达右端终止态比例。
    """
    success = 0
    for _ in range(n_trials):
        s = env.reset()
        for _ in range(max_steps):
            if env.is_terminal(s):
                break
            a = greedy_action(Q, s)
            s, r, done = env.step(s, a)
            if done:
                if s == env.terminal_right:
                    success += 1
                break
    return success / n_trials


def train_q_learning(env, gamma=0.95, alpha=0.1, epsilon=0.1, n_episodes=5000, max_steps=200, seed=0):
    """
    Q-learning 训练主循环：
    - 行为策略：ε-greedy 产生 (s,a,r,s')
    - 更新目标：r + gamma * max_{a'} Q(s',a')
    """
    rng = random.Random(seed)
    Q = defaultdict(float)

    checkpoints = [10, 50, 200, 1000, n_episodes]

    for ep in range(1, n_episodes + 1):
        s = env.reset()

        for _ in range(max_steps):
            if env.is_terminal(s):
                break

            a = epsilon_greedy_action(Q, s, epsilon, rng)
            s2, r, done = env.step(s, a)

            if done:
                td_target = r
            else:
                td_target = r + gamma * max(Q[(s2, LEFT)], Q[(s2, RIGHT)])

            td_error = td_target - Q[(s, a)]
            Q[(s, a)] += alpha * td_error

            s = s2
            if done:
                break

        if ep in checkpoints:
            eval_env = SlipperyRandomWalk(n_states=env.n_states, start_state=env.start_state,
                                          slip_prob=env.slip_prob, seed=999)
            sr = evaluate_greedy_policy(eval_env, Q, n_trials=300)
            print(f"episode={ep:>5} | greedy success_rate={sr:.3f} | epsilon={epsilon}")

    return Q


if __name__ == "__main__":
    env = SlipperyRandomWalk(n_states=7, start_state=3, slip_prob=0.2, seed=42)

    Q = train_q_learning(
        env,
        gamma=0.95,
        alpha=0.1,
        epsilon=0.1,
        n_episodes=5000,
        max_steps=200,
        seed=7
    )

    print("\nLearned greedy policy (non-terminal states):")
    for s in range(env.n_states):
        if env.is_terminal(s):
            continue
        a = greedy_action(Q, s)
        print(f"s={s}: {'RIGHT' if a==RIGHT else 'LEFT'} | QL={Q[(s,LEFT)]:.4f}, QR={Q[(s,RIGHT)]:.4f}")
