# -*- coding: utf-8 -*-
"""
lesson9_1_sarsa_tabular_control.py

整体在干什么？
1) 构造黑箱环境：1D Random Walk（可打滑），只能通过 reset/step 交互采样。
2) 用 SARSA（on-policy TD control）学习动作价值函数 Q(s,a)：
      Q(s,a) <- Q(s,a) + alpha * (r + gamma*Q(s',a') - Q(s,a))
   其中 a' 是在 s' 按当前 ε-greedy 策略采样得到的动作。
3) 策略使用 ε-greedy：
   - 以 ε 概率随机探索
   - 以 1-ε 概率选择 argmax_a Q(s,a)
4) 训练过程中周期性评估：从起点用贪心策略跑若干次，统计到达右端终止态的成功率。

你需要掌握：
- SARSA 更新使用 (S,A,R,S',A')，因此称 SARSA
- 它是 on-policy：目标动作 a' 来自同一条行为策略（ε-greedy）
- 学到 Q 后，用贪心/ε-greedy 即可实现策略改进
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

        # 只在到达右端终止态时给奖励
        r = 1.0 if s2 == self.terminal_right else 0.0
        return s2, r, done


def epsilon_greedy_action(Q, s: int, epsilon: float, rng: random.Random):
    """
    ε-greedy 选动作：
    - 以 ε 概率随机选 LEFT/RIGHT
    - 以 1-ε 概率选使 Q(s,a) 最大的动作（并列时偏向 RIGHT，稳定输出）
    """
    if rng.random() < epsilon:
        return rng.choice([LEFT, RIGHT])

    qL = Q[(s, LEFT)]
    qR = Q[(s, RIGHT)]
    return RIGHT if qR >= qL else LEFT


def greedy_action(Q, s: int):
    """
    纯贪心动作（用于评估，不探索）。
    """
    qL = Q[(s, LEFT)]
    qR = Q[(s, RIGHT)]
    return RIGHT if qR >= qL else LEFT


def evaluate_greedy_policy(env: SlipperyRandomWalk, Q, n_trials=200, max_steps=200):
    """
    评估：从起点出发，用贪心策略跑 n_trials 次，返回到达右端终止态的比例。
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


def train_sarsa(env, gamma=0.95, alpha=0.1, epsilon=0.1, n_episodes=5000, max_steps=200, seed=0):
    """
    SARSA 训练主循环：
    - 初始化 Q=0
    - 每条 episode：
        s0 = reset
        a0 = ε-greedy(Q,s0)
        while not done:
            s', r, done = step(s,a)
            a' = ε-greedy(Q,s')
            Q(s,a) <- Q(s,a) + alpha*(r + gamma*Q(s',a') - Q(s,a))
            s,a <- s',a'
    """
    rng = random.Random(seed)
    Q = defaultdict(float)

    # 终止态 Q 固定为0（不更新也没问题；这里不做特殊处理，逻辑更统一）
    checkpoints = [10, 50, 200, 1000, n_episodes]

    for ep in range(1, n_episodes + 1):
        s = env.reset()
        a = epsilon_greedy_action(Q, s, epsilon, rng)

        for _ in range(max_steps):
            s2, r, done = env.step(s, a)

            if done:
                # 终止后没有 a'，按惯例 target = r（因为 Q(terminal,·)=0）
                td_target = r
                td_error = td_target - Q[(s, a)]
                Q[(s, a)] += alpha * td_error
                break

            a2 = epsilon_greedy_action(Q, s2, epsilon, rng)

            td_target = r + gamma * Q[(s2, a2)]
            td_error = td_target - Q[(s, a)]
            Q[(s, a)] += alpha * td_error

            s, a = s2, a2

        if ep in checkpoints:
            eval_env = SlipperyRandomWalk(n_states=env.n_states, start_state=env.start_state,
                                          slip_prob=env.slip_prob, seed=999)  # 独立评估环境
            sr = evaluate_greedy_policy(eval_env, Q, n_trials=300)
            print(f"episode={ep:>5} | greedy success_rate={sr:.3f} | epsilon={epsilon}")

    return Q


if __name__ == "__main__":
    env = SlipperyRandomWalk(n_states=7, start_state=3, slip_prob=0.2, seed=42)

    Q = train_sarsa(
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
