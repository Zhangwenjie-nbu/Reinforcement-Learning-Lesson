# -*- coding: utf-8 -*-
"""
lesson7_3_policy_improvement_one_step.py

整体在干什么？
1) 构造一个小离散 MDP（状态0..4，终止态0和4，动作LEFT/RIGHT），模型(P,R)已知。
2) 指定一个初始策略 π（这里用“均匀随机策略”作为起点）。
3) 对 π 做策略评估，求 V^π（用贝尔曼期望迭代）。
4) 由 V^π 计算 Q^π(s,a)（一步展望）：
      Q^π(s,a) = Σ_{s'} P(s'|s,a) [ R + γ V^π(s') ]
5) 做一次策略改进得到 π'（确定性贪心）：
      π'(s) ∈ argmax_a Q^π(s,a)
6) 再对 π' 做策略评估得到 V^{π'}。
7) 比较 V^{π'} 与 V^π，验证策略改进定理的“不更差”现象。

你需要掌握：
- 策略改进的输入是 Q^π（或 V^π + 模型即可得到 Q^π）
- 输出是一条更贪心的策略 π'
- 在已知模型下你可以直接验证：V^{π'} >= V^π（数值上应成立）
"""

LEFT, RIGHT = 0, 1


def build_mdp_model():
    """
    构造小 MDP 模型 (P, R)：
    - 状态：0..4
    - 终止态：0 和 4
    - 动作：LEFT/RIGHT
    - 转移：确定性
    - 奖励：到达右端终止态4时给+1，否则0
    """
    states = list(range(5))
    terminal = {0, 4}
    actions = [LEFT, RIGHT]

    P = {}
    R = {}

    for s in states:
        for a in actions:
            if s in terminal:
                P[(s, a)] = {s: 1.0}
                R[(s, a, s)] = 0.0
                continue

            if a == LEFT:
                s2 = max(0, s - 1)
            else:
                s2 = min(4, s + 1)

            P[(s, a)] = {s2: 1.0}
            R[(s, a, s2)] = 1.0 if s2 == 4 else 0.0

    return states, terminal, actions, P, R


def pi_uniform_prob(s: int, a: int, terminal: set) -> float:
    """
    均匀随机策略 π(a|s)：
    - 非终止态：0.5
    - 终止态：动作无意义，返回0
    """
    if s in terminal:
        return 0.0
    return 0.5


def pi_deterministic_prob(policy_det, s: int, a: int, terminal: set) -> float:
    """
    确定性策略的概率形式：
    - 在非终止态：若 a==policy_det[s] 则 1 否则 0
    - 终止态：返回0
    """
    if s in terminal:
        return 0.0
    return 1.0 if policy_det[s] == a else 0.0


def bellman_expectation_backup_V(V, s: int, gamma: float, terminal: set, actions, P, R, policy_type, policy_det=None):
    """
    对单个状态做 V 的贝尔曼期望备份：
      V_new(s)=Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R + γ V(s')]

    policy_type:
    - "uniform": 使用均匀随机策略
    - "det": 使用确定性策略 policy_det
    """
    if s in terminal:
        return 0.0

    v_new = 0.0
    for a in actions:
        if policy_type == "uniform":
            pi = pi_uniform_prob(s, a, terminal)
        elif policy_type == "det":
            pi = pi_deterministic_prob(policy_det, s, a, terminal)
        else:
            raise ValueError("policy_type must be 'uniform' or 'det'.")

        if pi == 0.0:
            continue

        for s2, p in P[(s, a)].items():
            r = R[(s, a, s2)]
            v_new += pi * p * (r + gamma * V[s2])

    return v_new


def policy_evaluation_V(gamma, states, terminal, actions, P, R, policy_type, policy_det=None, tol=1e-12, max_iter=10000):
    """
    已知模型下策略评估（求 V^π）：
    - 迭代 V <- T^π V 直到收敛
    """
    V = [0.0 for _ in states]
    for _ in range(max_iter):
        V_new = V.copy()
        delta = 0.0
        for s in states:
            V_new[s] = bellman_expectation_backup_V(
                V, s, gamma, terminal, actions, P, R, policy_type, policy_det
            )
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < tol:
            break
    return V


def compute_Q_from_V(V, gamma, states, terminal, actions, P, R):
    """
    已知模型下，由 V^π 计算 Q^π(s,a)：
      Q^π(s,a)=Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ V^π(s') ]
    """
    Q = {(s, a): 0.0 for s in states for a in actions}
    for s in states:
        for a in actions:
            if s in terminal:
                Q[(s, a)] = 0.0
                continue
            q = 0.0
            for s2, p in P[(s, a)].items():
                r = R[(s, a, s2)]
                q += p * (r + gamma * V[s2])
            Q[(s, a)] = q
    return Q


def greedy_improve_policy_from_Q(Q, states, terminal):
    """
    根据 Q^π 做贪心策略改进，得到确定性策略 π'(s)=argmax_a Q(s,a)。
    并列时偏向 RIGHT，仅为输出稳定。
    """
    policy_det = {}
    for s in states:
        if s in terminal:
            continue
        qL = Q[(s, LEFT)]
        qR = Q[(s, RIGHT)]
        policy_det[s] = RIGHT if qR >= qL else LEFT
    return policy_det


if __name__ == "__main__":
    gamma = 0.95
    states, terminal, actions, P, R = build_mdp_model()

    # (1) 初始策略 π：均匀随机
    V_pi = policy_evaluation_V(gamma, states, terminal, actions, P, R, policy_type="uniform")
    Q_pi = compute_Q_from_V(V_pi, gamma, states, terminal, actions, P, R)

    # (2) 贪心改进得到 π'
    pi_prime_det = greedy_improve_policy_from_Q(Q_pi, states, terminal)

    # (3) 评估新策略 π'
    V_pi_prime = policy_evaluation_V(
        gamma, states, terminal, actions, P, R, policy_type="det", policy_det=pi_prime_det
    )

    print(f"gamma={gamma}\n")
    print("State | V^pi(s)     V^pi'(s)    improvement (V^pi' - V^pi)")
    print("------+-----------------------------------------------------")
    for s in states:
        imp = V_pi_prime[s] - V_pi[s]
        print(f"{s:>5} | {V_pi[s]:>10.6f}  {V_pi_prime[s]:>10.6f}  {imp:>14.6f}")

    print("\nGreedy improved policy π'(s) for non-terminal states:", pi_prime_det)

    print("\nQ^pi(s,a) used for improvement (non-terminal states):")
    for s in states:
        if s in terminal:
            continue
        print(f"s={s}: Q(LEFT)={Q_pi[(s, LEFT)]:.6f}, Q(RIGHT)={Q_pi[(s, RIGHT)]:.6f}")
