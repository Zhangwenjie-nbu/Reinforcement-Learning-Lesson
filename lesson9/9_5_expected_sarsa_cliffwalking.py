# -*- coding: utf-8 -*-
"""
lesson9_5_expected_sarsa_cliffwalking.py

整体在干什么？
1) 实现经典 CliffWalking（4x12）环境：
   - 每步 -1
   - 掉悬崖 -100 并回起点（episode 不终止）
   - 到达终点 done=True
2) 用同样的 ε-greedy 行为策略分别训练三种控制算法：
   - SARSA：target = r + gamma * Q(s', a')，a' ~ ε-greedy（on-policy）
   - Q-learning：target = r + gamma * max_a Q(s',a)（off-policy）
   - Expected SARSA：target = r + gamma * Σ_a π(a|s') Q(s',a)（on-policy，更稳定）
3) 输出滚动平均回报，用于比较探索风险场景下的差异。

你需要掌握：
- Expected SARSA 的关键：把“下一动作的采样值”替换成“对策略的期望”
- ε-greedy 下 π(a|s') 可直接由 Q 的贪心动作计算出来
"""

import random
from collections import defaultdict, deque

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTIONS = [UP, RIGHT, DOWN, LEFT]
ARROW = {UP: "↑", RIGHT: "→", DOWN: "↓", LEFT: "←"}


class CliffWalking:
    def __init__(self, n_rows=4, n_cols=12):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.start = (3, 0)
        self.goal = (3, 11)
        self.cliff = {(3, c) for c in range(1, 11)}

    def reset(self):
        return self._to_state(self.start)

    def step(self, s: int, a: int):
        r, c = self._from_state(s)

        if a == UP:
            r = max(0, r - 1)
        elif a == DOWN:
            r = min(self.n_rows - 1, r + 1)
        elif a == LEFT:
            c = max(0, c - 1)
        elif a == RIGHT:
            c = min(self.n_cols - 1, c + 1)

        nxt = (r, c)

        if nxt in self.cliff:
            return self._to_state(self.start), -100.0, False

        if nxt == self.goal:
            return self._to_state(nxt), -1.0, True

        return self._to_state(nxt), -1.0, False

    def _to_state(self, pos):
        r, c = pos
        return r * self.n_cols + c

    def _from_state(self, s: int):
        r = s // self.n_cols
        c = s % self.n_cols
        return (r, c)


def greedy_action(Q, s: int):
    best_a = ACTIONS[0]
    best_q = Q[(s, best_a)]
    for a in ACTIONS[1:]:
        q = Q[(s, a)]
        if q > best_q:
            best_q = q
            best_a = a
    return best_a


def epsilon_greedy_action(Q, s: int, epsilon: float, rng: random.Random):
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)
    return greedy_action(Q, s)


def expected_q_under_epsilon_greedy(Q, s: int, epsilon: float):
    """
    计算 Σ_a π(a|s) Q(s,a)，其中 π 为 ε-greedy（基于当前 Q 的贪心动作）。
    """
    nA = len(ACTIONS)
    a_g = greedy_action(Q, s)

    exp_q = 0.0
    for a in ACTIONS:
        if a == a_g:
            p = (1.0 - epsilon) + (epsilon / nA)
        else:
            p = epsilon / nA
        exp_q += p * Q[(s, a)]
    return exp_q


def train_sarsa(env, gamma, alpha, epsilon, n_episodes, seed=0, max_steps=10_000):
    rng = random.Random(seed)
    Q = defaultdict(float)
    returns = []

    for _ in range(n_episodes):
        s = env.reset()
        a = epsilon_greedy_action(Q, s, epsilon, rng)

        G = 0.0
        for _ in range(max_steps):
            s2, r, done = env.step(s, a)
            G += r

            if done:
                Q[(s, a)] += alpha * (r - Q[(s, a)])
                break

            a2 = epsilon_greedy_action(Q, s2, epsilon, rng)
            td_target = r + gamma * Q[(s2, a2)]
            Q[(s, a)] += alpha * (td_target - Q[(s, a)])

            s, a = s2, a2

        returns.append(G)

    return Q, returns


def train_q_learning(env, gamma, alpha, epsilon, n_episodes, seed=0, max_steps=10_000):
    rng = random.Random(seed)
    Q = defaultdict(float)
    returns = []

    for _ in range(n_episodes):
        s = env.reset()
        G = 0.0

        for _ in range(max_steps):
            a = epsilon_greedy_action(Q, s, epsilon, rng)
            s2, r, done = env.step(s, a)
            G += r

            if done:
                td_target = r
            else:
                td_target = r + gamma * max(Q[(s2, ap)] for ap in ACTIONS)

            Q[(s, a)] += alpha * (td_target - Q[(s, a)])
            s = s2

            if done:
                break

        returns.append(G)

    return Q, returns


def train_expected_sarsa(env, gamma, alpha, epsilon, n_episodes, seed=0, max_steps=10_000):
    rng = random.Random(seed)
    Q = defaultdict(float)
    returns = []

    for _ in range(n_episodes):
        s = env.reset()
        G = 0.0

        for _ in range(max_steps):
            a = epsilon_greedy_action(Q, s, epsilon, rng)
            s2, r, done = env.step(s, a)
            G += r

            if done:
                td_target = r
            else:
                td_target = r + gamma * expected_q_under_epsilon_greedy(Q, s2, epsilon)

            Q[(s, a)] += alpha * (td_target - Q[(s, a)])
            s = s2

            if done:
                break

        returns.append(G)

    return Q, returns


def print_rolling_average(label: str, returns, window=50):
    dq = deque(maxlen=window)
    for i, g in enumerate(returns, 1):
        dq.append(g)
        if i in (1, 10, 20, 50, 100, 200, len(returns)) or (i % 50 == 0):
            avg = sum(dq) / len(dq)
            print(f"{label}: episode={i:>4} | rolling_avg_return(window={len(dq)}) = {avg:>8.2f}")


if __name__ == "__main__":
    env = CliffWalking()

    gamma = 1.0
    alpha = 0.5
    epsilon = 0.1
    n_episodes = 500
    seed = 7

    Q_sarsa, ret_sarsa = train_sarsa(env, gamma, alpha, epsilon, n_episodes, seed=seed)
    Q_ql, ret_ql = train_q_learning(env, gamma, alpha, epsilon, n_episodes, seed=seed)
    Q_es, ret_es = train_expected_sarsa(env, gamma, alpha, epsilon, n_episodes, seed=seed)

    print("\n=== Rolling average returns (higher is better; less negative is better) ===")
    print_rolling_average("SARSA        ", ret_sarsa, window=50)
    print()
    print_rolling_average("Q-learning    ", ret_ql, window=50)
    print()
    print_rolling_average("Expected SARSA", ret_es, window=50)
