# -*- coding: utf-8 -*-
"""
lesson9_4_double_q_learning_overestimation_bias.py

整体在干什么？
1) 构造一个专门用于展示 Q-learning “maximization bias（过估计偏差）”的小型 episodic MDP：
   - 状态 A：动作 LEFT 直接终止奖励 0；动作 RIGHT 转到状态 B 奖励 0
   - 状态 B：有 K 个动作，每个动作终止且奖励 ~ N(0, 1)，期望为 0
   => 因而从 A 出发，LEFT 与 RIGHT 的真实期望都为 0（两者等价）

2) 分别训练：
   - Q-learning：target = r + gamma * max_a Q(s', a)
   - Double Q-learning：维护 Q1/Q2，选择与评估分离，缓解 max 的过估计

3) 记录训练过程中在状态 A 选择 LEFT 的比例（越接近 0.5 越合理）：
   - 若 Q-learning 产生过估计，会更偏向选择 RIGHT，使 LEFT 比例 < 0.5
   - Double Q-learning 应更接近 0.5

你需要掌握：
- “max” 在含噪估计上会系统性偏大
- Double Q 的关键：用 Q1 做 argmax，用 Q2 做 evaluation（或反过来），降低过估计
"""

import random
from collections import defaultdict, deque

A_STATE = 0
B_STATE = 1
TERMINAL = 2

LEFT = 0
RIGHT = 1


class MaximizationBiasMDP:
    """
    Sutton 风格的 maximization-bias 演示环境：
    - A: actions {LEFT, RIGHT}
    - LEFT -> terminal, reward 0
    - RIGHT -> B, reward 0
    - B: actions {0..K-1}
      each -> terminal, reward ~ Normal(0, 1)
    """

    def __init__(self, n_actions_b=10, seed=0):
        self.n_actions_b = n_actions_b
        self.rng = random.Random(seed)

    def reset(self):
        """重置到状态 A。"""
        return A_STATE

    def actions(self, s: int):
        """返回当前状态可用动作集合。"""
        if s == A_STATE:
            return [LEFT, RIGHT]
        if s == B_STATE:
            return list(range(self.n_actions_b))
        return []

    def step(self, s: int, a: int):
        """执行一步，返回 (s2, r, done)。"""
        if s == A_STATE:
            if a == LEFT:
                return TERMINAL, 0.0, True
            else:
                return B_STATE, 0.0, False

        if s == B_STATE:
            # 终止并给噪声奖励，期望为 0
            r = self.rng.gauss(0.0, 1.0)
            return TERMINAL, r, True

        return TERMINAL, 0.0, True


# def argmax_action(Q, s: int, actions):
#     """
#     选取使 Q(s,a) 最大的动作（并列时取动作编号最小者，保证确定性）。
#     """
#     best_a = actions[0]
#     best_q = Q[(s, best_a)]
#     for a in actions[1:]:
#         q = Q[(s, a)]
#         if q > best_q:
#             best_q = q
#             best_a = a
#     return best_a

def argmax_action(Q, s: int, actions, rng: random.Random):
    best_q = None
    best_actions = []
    for a in actions:
        q = Q[(s, a)]
        if (best_q is None) or (q > best_q):
            best_q = q
            best_actions = [a]
        elif q == best_q:
            best_actions.append(a)
    return rng.choice(best_actions)



def epsilon_greedy(Q, s: int, actions, epsilon: float, rng: random.Random):
    """
    ε-greedy 行为策略：
    - ε 概率随机
    - 1-ε 概率 argmax
    """
    if rng.random() < epsilon:
        return rng.choice(actions)
    return argmax_action(Q, s, actions, rng = rng)


def train_q_learning(env: MaximizationBiasMDP, gamma=1.0, alpha=0.1, epsilon=0.1, n_episodes=50_000, seed=0):
    """
    Q-learning 训练：
    - 行为策略：ε-greedy
    - 更新目标：r + gamma * max_a Q(s',a)
    返回：
    - Q
    - left_rate_history: 若干 checkpoint 上在 A 选 LEFT 的累计比例
    """
    rng = random.Random(seed)
    Q = defaultdict(float)

    left_count = 0
    history = []

    checkpoints = [100, 500, 1_000, 5_000, 10_000, 20_000, n_episodes]

    for ep in range(1, n_episodes + 1):
        s = env.reset()

        # 第一步必在 A，记录动作偏好
        a = epsilon_greedy(Q, s, env.actions(s), epsilon, rng)
        if a == LEFT:
            left_count += 1

        s2, r, done = env.step(s, a)

        if done:
            target = r
            Q[(s, a)] += alpha * (target - Q[(s, a)])
        else:
            # 到了 B，再走一步终止
            # 在 Q-learning 中，我们仍用 ε-greedy 产生行为，但 target 用 max
            a2 = epsilon_greedy(Q, s2, env.actions(s2), epsilon, rng)
            s3, r2, done2 = env.step(s2, a2)

            # 更新 B 的 Q
            target_b = r2  # terminal
            Q[(s2, a2)] += alpha * (target_b - Q[(s2, a2)])

            # 更新 A 的 Q：target = 0 + gamma*max_a Q(B,a)
            max_q_b = max(Q[(B_STATE, ab)] for ab in env.actions(B_STATE))
            target_a = 0.0 + gamma * max_q_b
            Q[(s, a)] += alpha * (target_a - Q[(s, a)])

        if ep in checkpoints:
            history.append((ep, left_count / ep))

    return Q, history


def train_double_q_learning(env: MaximizationBiasMDP, gamma=1.0, alpha=0.1, epsilon=0.1, n_episodes=50_000, seed=0):
    """
    Double Q-learning 训练：
    - 维护 Q1, Q2
    - 行为策略通常用 Q1+Q2 做 ε-greedy（更稳定）
    - 每次随机更新其中一张表：
        若更新 Q1:
           a* = argmax_a Q1(s',a)
           target = r + gamma * Q2(s', a*)
        若更新 Q2: 对称交换
    返回：
    - Q1, Q2
    - history: 若干 checkpoint 上在 A 选 LEFT 的累计比例
    """
    rng = random.Random(seed)
    Q1 = defaultdict(float)
    Q2 = defaultdict(float)

    left_count = 0
    history = []
    checkpoints = [100, 500, 1_000, 5_000, 10_000, 20_000, n_episodes]

    def Qsum(sa):
        s, a = sa
        return Q1[(s, a)] + Q2[(s, a)]

    for ep in range(1, n_episodes + 1):
        s = env.reset()

        # A 上动作选择基于 Q1+Q2（行为策略）
        actions_a = env.actions(s)
        if rng.random() < epsilon:
            a = rng.choice(actions_a)
        else:
            a = argmax_action({(s, aa): Qsum((s, aa)) for aa in actions_a}, s, actions_a, rng = rng)

        if a == LEFT:
            left_count += 1

        s2, r, done = env.step(s, a)

        if done:
            # 终止：只更新一次
            if rng.random() < 0.5:
                Q1[(s, a)] += alpha * (r - Q1[(s, a)])
            else:
                Q2[(s, a)] += alpha * (r - Q2[(s, a)])
        else:
            # 到了 B，再走一步终止
            actions_b = env.actions(s2)
            # 行为动作仍基于 Q1+Q2
            if rng.random() < epsilon:
                a2 = rng.choice(actions_b)
            else:
                a2 = argmax_action({(s2, ab): Qsum((s2, ab)) for ab in actions_b}, s2, actions_b, rng = rng)

            s3, r2, done2 = env.step(s2, a2)

            # 先更新 B 的一步（终止）
            if rng.random() < 0.5:
                Q1[(s2, a2)] += alpha * (r2 - Q1[(s2, a2)])
            else:
                Q2[(s2, a2)] += alpha * (r2 - Q2[(s2, a2)])

            # 再更新 A 的 Q（关键：选择与评估分离）
            if rng.random() < 0.5:
                # 更新 Q1，用 Q1 选 a*，用 Q2 评估
                a_star = argmax_action(Q1, B_STATE, actions_b, rng = rng)
                target = 0.0 + gamma * Q2[(B_STATE, a_star)]
                Q1[(s, a)] += alpha * (target - Q1[(s, a)])
            else:
                # 更新 Q2，用 Q2 选 a*，用 Q1 评估
                a_star = argmax_action(Q2, B_STATE, actions_b, rng = rng)
                target = 0.0 + gamma * Q1[(B_STATE, a_star)]
                Q2[(s, a)] += alpha * (target - Q2[(s, a)])

        if ep in checkpoints:
            history.append((ep, left_count / ep))

    return Q1, Q2, history


def print_history(title: str, hist):
    """打印若干 checkpoint 上 LEFT 比例。"""
    print(title)
    for ep, p in hist:
        print(f"  episode={ep:>7} | P(choose LEFT at A)={p:.3f}")
    print()


if __name__ == "__main__":
    env = MaximizationBiasMDP(n_actions_b=10, seed=42)

    gamma = 1.0
    alpha = 0.1
    epsilon = 0.1
    n_episodes = 50_000
    seed = 7

    Q, hist_q = train_q_learning(env, gamma=gamma, alpha=alpha, epsilon=epsilon, n_episodes=n_episodes, seed=seed)
    Q1, Q2, hist_dq = train_double_q_learning(env, gamma=gamma, alpha=alpha, epsilon=epsilon, n_episodes=n_episodes, seed=seed)

    print_history("Q-learning history:", hist_q)
    print_history("Double Q-learning history:", hist_dq)

    # 最终 A 上两个动作的估计值对比
    q_left = Q[(A_STATE, LEFT)]
    q_right = Q[(A_STATE, RIGHT)]
    dq_left = Q1[(A_STATE, LEFT)] + Q2[(A_STATE, LEFT)]
    dq_right = Q1[(A_STATE, RIGHT)] + Q2[(A_STATE, RIGHT)]

    print("Final Q(A,LEFT) vs Q(A,RIGHT):")
    print(f"  Q-learning : LEFT={q_left:.4f} | RIGHT={q_right:.4f}")
    print(f"  Double-Q   : LEFT={dq_left:.4f} | RIGHT={dq_right:.4f}")
