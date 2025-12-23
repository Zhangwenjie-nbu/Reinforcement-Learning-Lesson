# -*- coding: utf-8 -*-
"""
lesson6_2_bellman_expectation_q_policy_evaluation.py

整体在干什么？
1) 构造一个小离散 MDP（与 6.1 相同）：状态0..4，终止态0和4，动作LEFT/RIGHT。
2) 给定固定策略 π（均匀随机：LEFT/RIGHT 各0.5）。
3) 显式写出 MDP 模型：P(s'|s,a) 与 R(s,a,s')。
4) 用 Q 的贝尔曼期望方程做策略评估（policy evaluation）：
      Q_{k+1}(s,a) = Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ Σ_{a'} π(a'|s') Q_k(s',a') ]
   迭代直到收敛。
5) 用收敛后的 Q 通过 V(s)=Σ_a π(a|s)Q(s,a) 得到 V，并打印出来以验证关系。

你需要掌握：
- Q 的贝尔曼期望方程比 V 的版本多了一层“在 s' 处按策略对 a' 求期望”
- 这是定义层面的必然结果（先一步，再继续按策略）
- 通过 Q 可以恢复 V（按策略加权）
"""

LEFT, RIGHT = 0, 1


def build_mdp_model():
    """
    构造小 MDP 的模型 (P, R)：
    - 状态：0..4
    - 终止态：0 和 4
    - 动作：LEFT/RIGHT
    - 转移：确定性（边界夹紧到终止态）
    - 奖励：到达右端终止态4时给+1，否则0

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
                # 终止态：吸收态建模，保持在原地，奖励为0
                P[(s, a)] = {s: 1.0}
                R[(s, a, s)] = 0.0
                continue

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
    - 非终止态：LEFT/RIGHT 各0.5
    - 终止态：动作无意义，返回0（不会影响结果）
    """
    if s in terminal:
        return 0.0
    return 0.5


def expected_Q_under_policy(Q, s: int, terminal: set, actions):
    """
    计算 Σ_{a'} π(a'|s) Q(s,a')，即在状态 s 下按策略对 Q 做动作期望。

    逻辑：
    - 终止态的 V 视为0（回报结束/吸收态奖励为0）
    - 非终止态：用均匀随机策略的0.5权重做加权
    """
    if s in terminal:
        return 0.0

    v = 0.0
    for a in actions:
        pi = uniform_random_policy_prob(s, a, terminal)
        v += pi * Q[(s, a)]
    return v


def bellman_expectation_backup_Q(Q, s: int, a: int, gamma: float, terminal: set, actions, P, R):
    """
    对单个 (s,a) 做一次 Q 的贝尔曼期望备份：
      (T^π_Q Q)(s,a) = Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ Σ_{a'} π(a'|s') Q(s',a') ]

    逻辑：
    - 终止态的 (s,a)：Q=0（吸收态、无后续回报）
    - 非终止态：先对 s' 求期望，再加上 s' 的按策略动作期望
    """
    if s in terminal:
        return 0.0

    q_new = 0.0
    for s2, p in P[(s, a)].items():
        r = R[(s, a, s2)]
        v_next = expected_Q_under_policy(Q, s2, terminal, actions)
        q_new += p * (r + gamma * v_next)

    return q_new


def policy_evaluation_Q(gamma: float = 0.95, tol: float = 1e-10, max_iter: int = 10000):
    """
    用迭代法做 Q 的策略评估：
    - 初始化 Q=0
    - 重复对所有 (s,a) 做贝尔曼期望备份，直到收敛（最大变化 < tol）

    返回：
    - Q: dict[(s,a)] -> float
    - V_from_Q: list[float]，由 Q 恢复的 V(s)=Σ_a π(a|s)Q(s,a)
    """
    states, terminal, actions, P, R = build_mdp_model()

    Q = {(s, a): 0.0 for s in states for a in actions}

    for it in range(max_iter):
        delta = 0.0
        Q_new = Q.copy()

        for s in states:
            for a in actions:
                Q_new[(s, a)] = bellman_expectation_backup_Q(Q, s, a, gamma, terminal, actions, P, R)
                delta = max(delta, abs(Q_new[(s, a)] - Q[(s, a)]))

        Q = Q_new

        # 打印前几轮与收敛时信息，避免输出过多
        if it < 10 or delta < tol:
            # 仅打印每个状态的 V（由 Q 恢复）以更易读
            V_from_Q = [expected_Q_under_policy(Q, s, terminal, actions) for s in states]
            print(f"iter={it:>4} | delta={delta:.3e} | V_from_Q={['{:.6f}'.format(x) for x in V_from_Q]}")

        if delta < tol:
            break

    V_from_Q = [expected_Q_under_policy(Q, s, terminal, actions) for s in states]
    return Q, V_from_Q


if __name__ == "__main__":
    Q, V_from_Q = policy_evaluation_Q(gamma=0.95, tol=1e-12, max_iter=10000)

    print("\nFinal V reconstructed from Q:", [round(v, 6) for v in V_from_Q])

    # 也打印最终 Q，便于你观察“动作层面”价值差异
    print("\nFinal Q(s,a):")
    for s in range(5):
        print(f"s={s}: Q(LEFT)={Q[(s, LEFT)]:.6f}, Q(RIGHT)={Q[(s, RIGHT)]:.6f}")
