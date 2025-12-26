# -*- coding: utf-8 -*-
"""
lesson9_3_cliffwalking_sarsa_vs_qlearning.py

整体在干什么？
1) 实现经典 CliffWalking 环境（4x12 网格）：
   - 每步奖励 -1
   - 掉入悬崖奖励 -100，并被送回起点（episode 不结束）
   - 到达终点 episode 结束
2) 用同样的 ε-greedy 行为策略分别训练：
   - SARSA（on-policy）：target = r + gamma * Q(s', a')，a' 来自 ε-greedy
   - Q-learning（off-policy）：target = r + gamma * max_{a'} Q(s', a')
3) 打印训练过程中的平均回报（rolling average），比较两者在持续探索下的典型差异。
4) 打印最终贪心策略的箭头图，观察 SARSA 更安全、Q-learning 更贴悬崖的现象。

你需要掌握：
- 差异只来自目标项：SARSA 用采样到的 a'，Q-learning 用 max
- 在固定 ε 探索下，SARSA 往往学出对探索更鲁棒的安全路线，训练回报更好
- Q-learning 学的是最优贪心路线，但在训练时行为仍探索，容易反复掉崖导致回报更差
"""

import random
from collections import defaultdict, deque

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTIONS = [UP, RIGHT, DOWN, LEFT]
ARROW = {UP: "↑", RIGHT: "→", DOWN: "↓", LEFT: "←"}


class CliffWalking:
    """
    经典 4x12 CliffWalking（参考 Sutton & Barto）：
    - 网格坐标：(row, col)，row=0 在上，row=3 在下
    - 起点：S=(3,0)
    - 终点：G=(3,11)
    - 悬崖：row=3 且 col=1..10
    动力学：
    - 普通移动：每步 reward=-1
    - 若下一格落入悬崖：reward=-100，并回到起点（episode 不终止）
    - 到达终点：reward=-1，并 done=True
    """

    def __init__(self, n_rows=4, n_cols=12):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.start = (3, 0)
        self.goal = (3, 11)
        self.cliff = {(3, c) for c in range(1, 11)}

    def reset(self):
        """
        重置到起点，返回状态（用整数编码）。
        """
        return self._to_state(self.start)

    def step(self, s: int, a: int):
        """
        执行动作，返回 (s2, r, done)。
        """
        pos = self._from_state(s)
        nr, nc = pos

        if a == UP:
            nr = max(0, nr - 1)
        elif a == DOWN:
            nr = min(self.n_rows - 1, nr + 1)
        elif a == LEFT:
            nc = max(0, nc - 1)
        elif a == RIGHT:
            nc = min(self.n_cols - 1, nc + 1)

        next_pos = (nr, nc)

        # 掉悬崖：强惩罚并回起点，不结束
        if next_pos in self.cliff:
            return self._to_state(self.start), -100.0, False

        # 到终点：结束
        if next_pos == self.goal:
            return self._to_state(next_pos), -1.0, True

        # 普通一步
        return self._to_state(next_pos), -1.0, False

    def _to_state(self, pos):
        """
        (row,col) -> state_id
        """
        r, c = pos
        return r * self.n_cols + c

    def _from_state(self, s: int):
        """
        state_id -> (row,col)
        """
        r = s // self.n_cols
        c = s % self.n_cols
        return (r, c)

    def render_greedy_policy(self, Q):
        """
        打印贪心策略（按 Q 最大动作），悬崖标记为 '####'，起点 S，终点 G。
        """
        lines = []
        for r in range(self.n_rows):
            row_chars = []
            for c in range(self.n_cols):
                pos = (r, c)
                if pos == self.start:
                    row_chars.append(" S ")
                elif pos == self.goal:
                    row_chars.append(" G ")
                elif pos in self.cliff:
                    row_chars.append("###")
                else:
                    s = self._to_state(pos)
                    a = greedy_action(Q, s)
                    row_chars.append(f" {ARROW[a]} ")
            lines.append("".join(row_chars))
        return "\n".join(lines)


def greedy_action(Q, s: int):
    """
    纯贪心动作（并列时按固定顺序选，保证输出稳定）。
    """
    best_a = ACTIONS[0]
    best_q = Q[(s, best_a)]
    for a in ACTIONS[1:]:
        q = Q[(s, a)]
        if q > best_q:
            best_q = q
            best_a = a
    return best_a


def epsilon_greedy_action(Q, s: int, epsilon: float, rng: random.Random):
    """
    ε-greedy 行为策略：
    - ε 概率随机动作
    - 1-ε 概率贪心动作
    """
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)
    return greedy_action(Q, s)


def train_sarsa(env: CliffWalking, gamma=1.0, alpha=0.5, epsilon=0.1, n_episodes=500, seed=0, max_steps=10_000):
    """
    SARSA 训练：
    - 行为策略：ε-greedy
    - 更新目标：r + gamma*Q(s',a')，其中 a' 由 ε-greedy 产生（on-policy）
    返回：
    - Q
    - returns: 每个 episode 的总回报（越大越好，负得越少越好）
    """
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
                td_target = r  # 终点后没有后续价值
                Q[(s, a)] += alpha * (td_target - Q[(s, a)])
                break

            a2 = epsilon_greedy_action(Q, s2, epsilon, rng)
            td_target = r + gamma * Q[(s2, a2)]
            Q[(s, a)] += alpha * (td_target - Q[(s, a)])

            s, a = s2, a2

        returns.append(G)

    return Q, returns


def train_q_learning(env: CliffWalking, gamma=1.0, alpha=0.5, epsilon=0.1, n_episodes=500, seed=0, max_steps=10_000):
    """
    Q-learning 训练：
    - 行为策略：ε-greedy（用于采样数据）
    - 更新目标：r + gamma*max_a' Q(s',a')（off-policy，目标策略为贪心）
    返回：
    - Q
    - returns: 每个 episode 的总回报
    """
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


def print_rolling_average(label: str, returns, window=50):
    """
    打印滚动平均回报，便于比较学习曲线的趋势。
    """
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

    Q_sarsa, ret_sarsa = train_sarsa(env, gamma=gamma, alpha=alpha, epsilon=epsilon, n_episodes=n_episodes, seed=seed)
    Q_ql, ret_ql = train_q_learning(env, gamma=gamma, alpha=alpha, epsilon=epsilon, n_episodes=n_episodes, seed=seed)

    print("\n=== Rolling average returns (higher is better; less negative means fewer cliff falls / shorter paths) ===")
    print_rolling_average("SARSA    ", ret_sarsa, window=50)
    print()
    print_rolling_average("Q-learning", ret_ql, window=50)

    print("\n=== Greedy policy learned by SARSA (arrows) ===")
    print(env.render_greedy_policy(Q_sarsa))

    print("\n=== Greedy policy learned by Q-learning (arrows) ===")
    print(env.render_greedy_policy(Q_ql))
