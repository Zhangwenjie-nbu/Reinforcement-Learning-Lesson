# -*- coding: utf-8 -*-
"""
lesson6_1_bellman_expectation_policy_evaluation.py

整体在干什么？
1) 构造一个很小的离散 MDP（5个状态：0..4），其中 0 和 4 是终止态。
2) 指定一个固定策略 π（均匀随机：LEFT/RIGHT 各 0.5）。
3) 明确写出该 MDP 的模型：
   - 转移概率 P(s'|s,a)
   - 奖励函数 R(s,a,s')
4) 用贝尔曼期望方程对应的“策略评估迭代”（policy evaluation）计算 V^π：
      V_{k+1}(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ V_k(s') ]
   重复迭代直到收敛，并打印每轮的 V 向量变化。

你需要掌握：
- 这段代码实现的不是学习（learning），而是“已知模型下的计算”（DP / policy evaluation）
- 迭代更新正是贝尔曼期望方程的算子形式
- 终止态的处理：终止态价值固定为 0（吸收/回报结束）
"""

import math

LEFT, RIGHT = 0, 1


def build_mdp_model():
    """
    构造一个小 MDP 的模型 (P, R)：
    - 状态：0..4
    - 终止态：0 和 4
    - 动作：LEFT/RIGHT
    - 转移：确定性（边界夹紧到终止态）
    - 奖励：到达右端终止态 4 时给 +1，否则 0

    返回：
    - states: list[int]
    - terminal: set[int]
    - actions: list[int]
    - P: dict[(s,a)] -> dict[s'] = prob
    - R: dict[(s,a,s')] -> reward
    """
    states = list(range(5))
    terminal = {0, 4}
    actions = [LEFT, RIGHT]

    P = {}
    R = {}

    for s in states:
        for a in actions:
            if s in terminal:
                # 终止态：保持在原地（吸收态建模），奖励为0
                P[(s, a)] = {s: 1.0}
                R[(s, a, s)] = 0.0
                continue

            # 非终止态：确定性移动
            if a == LEFT:
                s2 = max(0, s - 1)
            else:
                s2 = min(4, s + 1)

            P[(s, a)] = {s2: 1.0}
            reward = 1.0 if s2 == 4 else 0.0
            R[(s, a, s2)] = reward

    return states, terminal, actions, P, R


def uniform_random_policy_prob(s: int, a: int, terminal: set):
    """
    均匀随机策略 π(a|s)：
    - 在非终止态：LEFT/RIGHT 各 0.5
    - 在终止态：动作无意义，这里返回 0（不会被用到或影响结果）
    """
    if s in terminal:
        return 0.0
    return 0.5


def bellman_expectation_backup(V, s: int, gamma: float, terminal: set, actions, P, R):
    """
    对单个状态 s 做一次贝尔曼期望备份（Bellman expectation backup）：
    计算：
      (T^π V)(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ V(s') ]

    逻辑：
    - 终止态价值固定为0（也可以直接返回 V[s]，但固定0更清晰）
    - 非终止态：按策略与转移做两层期望
    """
    if s in terminal:
        return 0.0

    v_new = 0.0
    for a in actions:
        pi = uniform_random_policy_prob(s, a, terminal)
        for s2, p in P[(s, a)].items():
            r = R[(s, a, s2)]
            v_new += pi * p * (r + gamma * V[s2])

    return v_new


def policy_evaluation(gamma: float = 0.95, tol: float = 1e-10, max_iter: int = 10000):
    """
    用迭代法做策略评估：
    - 初始化 V=0
    - 重复应用 T^π 直到收敛（最大变化 < tol）或达到 max_iter

    返回：
    - V: list[float]，每个状态的估计价值
    """
    states, terminal, actions, P, R = build_mdp_model()
    V = [0.0 for _ in states]

    for it in range(max_iter):
        delta = 0.0
        V_new = V.copy()

        for s in states:
            V_new[s] = bellman_expectation_backup(V, s, gamma, terminal, actions, P, R)
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new

        # 打印前几轮与收敛时的变化（避免输出太多）
        if it < 10 or delta < tol:
            print(f"iter={it:>4} | delta={delta:.3e} | V={['{:.6f}'.format(x) for x in V]}")

        if delta < tol:
            break

    return V


if __name__ == "__main__":
    V = policy_evaluation(gamma=0.95, tol=1e-12, max_iter=10000)
    print("\nFinal V:", [round(v, 6) for v in V])
