# -*- coding: utf-8 -*-
"""
lesson7_2_bellman_optimality_value_iteration.py

整体在干什么？
1) 构造一个小离散 MDP（状态0..4，终止态0和4，动作LEFT/RIGHT），模型(P,R)已知。
2) 实现贝尔曼最优算子 T*：
      (T* V)(s) = max_a Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ V(s') ]
3) 用 Value Iteration 迭代求解最优价值函数 V*：
      V_{k+1} = T* V_k
   直到收敛。
4) 收敛后从 V* 导出最优 Q*（一阶段展望）与贪心最优策略：
      Q*(s,a) = Σ_{s'} P(s'|s,a) [ R + γ V*(s') ]
      π*(s) ∈ argmax_a Q*(s,a)

你需要掌握：
- Value Iteration 对应的就是贝尔曼最优方程的算子形式
- 终止态价值固定为0（回报结束/吸收态）
- 从 V* 可以直接导出最优动作（贪心）
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


def bellman_optimality_backup(V, s: int, gamma: float, terminal: set, actions, P, R):
    """
    对单个状态 s 做一次贝尔曼最优备份（Bellman optimality backup）：
      (T* V)(s) = max_a Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ V(s') ]

    逻辑：
    - 终止态固定为0
    - 非终止态：对每个动作做一步展望，取最大
    """
    if s in terminal:
        return 0.0

    best = None
    for a in actions:
        q = 0.0
        for s2, p in P[(s, a)].items():
            r = R[(s, a, s2)]
            q += p * (r + gamma * V[s2])
        best = q if (best is None or q > best) else best

    return best


def value_iteration(gamma: float = 0.95, tol: float = 1e-12, max_iter: int = 10000):
    """
    Value Iteration：
    - 初始化 V=0
    - 反复应用最优算子 T*，直到收敛

    返回：
    - V_star: list[float]
    - Q_star: dict[(s,a)] -> float（由 V_star 做一步展望得到）
    - greedy_policy: dict[s] -> a（对 Q_star 贪心）
    """
    states, terminal, actions, P, R = build_mdp_model()
    V = [0.0 for _ in states]

    for it in range(max_iter):
        delta = 0.0
        V_new = V.copy()

        for s in states:
            V_new[s] = bellman_optimality_backup(V, s, gamma, terminal, actions, P, R)
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new

        # 打印前几轮与收敛时信息
        if it < 10 or delta < tol:
            print(f"iter={it:>4} | delta={delta:.3e} | V={['{:.6f}'.format(x) for x in V]}")

        if delta < tol:
            break

    # 由 V* 计算 Q*（一步展望）
    Q = {}
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

    # 贪心策略（并列时偏向 RIGHT 只是为了输出稳定）
    greedy = {}
    for s in states:
        if s in terminal:
            continue
        qL = Q[(s, LEFT)]
        qR = Q[(s, RIGHT)]
        greedy[s] = RIGHT if qR >= qL else LEFT

    return V, Q, greedy


if __name__ == "__main__":
    gamma = 0.95
    V_star, Q_star, pi_star = value_iteration(gamma=gamma, tol=1e-12, max_iter=10000)

    print("\nFinal V*:", [round(v, 6) for v in V_star])

    print("\nFinal Q*(s,a) derived from V*:")
    for s in range(5):
        print(f"s={s}: Q(LEFT)={Q_star[(s, LEFT)]:.6f}, Q(RIGHT)={Q_star[(s, RIGHT)]:.6f}")

    print("\nGreedy optimal policy π*(s) on non-terminal states:", pi_star)
