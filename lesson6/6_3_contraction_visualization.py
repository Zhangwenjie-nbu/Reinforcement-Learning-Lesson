# -*- coding: utf-8 -*-
"""
lesson6_3_contraction_visualization.py

整体在干什么？
1) 复用一个小离散 MDP（状态0..4，终止态0和4）与固定策略 π（均匀随机）。
2) 实现贝尔曼期望算子 T^π（对 V 做一次备份得到新 V）。
3) 从两个截然不同的初始价值函数开始迭代：
   - V_A 初值全 0
   - V_B 初值全 100
4) 每轮迭代后计算两者差异的无穷范数：
      d_k = ||V_A - V_B||_∞
   并打印 d_k 以及 d_{k+1}/d_k 的比值（应接近 gamma）。
5) 直观看到“收缩”：差距每轮被压缩到约 gamma 倍，从而两条序列被拉向同一个固定点 V^π。

你需要掌握：
- 收敛不是“碰巧”，而是因为算子对误差具有收缩性（gamma<1）
- 这解释了策略评估迭代为什么从任意初值都能收敛到同一解
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


def pi_uniform(s: int, a: int, terminal: set) -> float:
    """
    均匀随机策略 π(a|s)：
    - 非终止态：0.5
    - 终止态：动作无意义，返回0
    """
    if s in terminal:
        return 0.0
    return 0.5


def bellman_T_pi(V, gamma: float, states, terminal, actions, P, R):
    """
    对整个 V 向量应用一次贝尔曼期望算子 T^π，返回新向量 V_new。

    逻辑：
    - 终止态价值固定为 0
    - 非终止态：V_new(s)=Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R + γ V(s')]
    """
    V_new = V.copy()

    for s in states:
        if s in terminal:
            V_new[s] = 0.0
            continue

        v = 0.0
        for a in actions:
            pi = pi_uniform(s, a, terminal)
            for s2, p in P[(s, a)].items():
                r = R[(s, a, s2)]
                v += pi * p * (r + gamma * V[s2])
        V_new[s] = v

    return V_new


def linf_norm_diff(V1, V2):
    """
    计算无穷范数差 ||V1 - V2||_∞ = max_s |V1(s)-V2(s)|
    """
    return max(abs(a - b) for a, b in zip(V1, V2))


if __name__ == "__main__":
    gamma = 0.95
    states, terminal, actions, P, R = build_mdp_model()

    # 两个截然不同的初值
    V_A = [0.0 for _ in states]
    V_B = [100.0 for _ in states]

    # 终止态固定为0（确保比较纯粹）
    for s in terminal:
        V_A[s] = 0.0
        V_B[s] = 0.0

    print(f"gamma = {gamma}\n")
    d_prev = linf_norm_diff(V_A, V_B)
    print(f"iter=0   | d0=||V_A - V_B||_∞ = {d_prev:.6f}")

    for it in range(1, 21):
        V_A = bellman_T_pi(V_A, gamma, states, terminal, actions, P, R)
        V_B = bellman_T_pi(V_B, gamma, states, terminal, actions, P, R)

        d = linf_norm_diff(V_A, V_B)
        ratio = (d / d_prev) if d_prev > 0 else float("nan")

        print(f"iter={it:<2} | d={d:.6f} | d/d_prev={ratio:.6f}")
        d_prev = d

    print("\nFinal V_A:", [round(v, 6) for v in V_A])
    print("Final V_B:", [round(v, 6) for v in V_B])
    print("Final ||V_A - V_B||_∞:", linf_norm_diff(V_A, V_B))
