# -*- coding: utf-8 -*-
"""
lesson7_1_optimality_by_enumerating_policies.py

整体在干什么？
1) 构造一个小离散 MDP（状态0..4，终止态0和4，动作LEFT/RIGHT），模型(P,R)已知。
2) 枚举所有“确定性平稳策略”（对每个非终止状态指定一个动作）：
   - 非终止状态是 {1,2,3}，每个有2个动作，因此共有 2^3 = 8 个策略。
3) 对每个策略 π：
   - 用贝尔曼期望算子的策略评估迭代求 V^π（已知模型，DP计算，不是学习）。
4) 对每个状态 s：
   - 取所有策略下 V^π(s) 的最大值，得到 V*(s)（按定义 max_π）。
5) 再按定义构造 Q*(s,a)：
   - Q*(s,a) = max_π Q^π(s,a)
   - 对给定 (s,a)，枚举“后续策略”并计算该策略下的 Q^π(s,a)
6) 验证关系：
   - V*(s) 是否等于 max_a Q*(s,a)
   - 并给出由 Q* 导出的贪心最优动作

你需要掌握：
- 这是“定义式求解”的最原始版本：通过枚举策略来取 max
- 小问题上它能工作，大问题上它不可行（策略数爆炸），因此我们才需要贝尔曼最优方程与学习算法
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


def make_policy_from_bits(bits):
    """
    将 bits（长度3的0/1列表）转换为一个确定性策略：
    - 对状态1、2、3分别指定动作
    - bit=0 -> LEFT, bit=1 -> RIGHT

    返回 policy: dict[s] -> a
    """
    mapping_states = [1, 2, 3]
    policy = {}
    for s, b in zip(mapping_states, bits):
        policy[s] = RIGHT if b == 1 else LEFT
    return policy


def enumerate_deterministic_policies():
    """
    枚举所有确定性平稳策略（这里只对非终止态1,2,3做决定）：
    共 2^3 = 8 个。
    """
    policies = []
    for b1 in [0, 1]:
        for b2 in [0, 1]:
            for b3 in [0, 1]:
                policies.append(make_policy_from_bits([b1, b2, b3]))
    return policies


def bellman_T_pi(V, policy, gamma, states, terminal, actions, P, R):
    """
    对给定策略 policy 应用一次贝尔曼期望算子 T^π，返回 V_new。
    这里策略是确定性的：π(a|s)=1 对 a=policy[s]，否则0。

    逻辑：
    - 终止态 V=0
    - 非终止态：V(s)=Σ_{s'} P(s'|s,a) [R + γ V(s')]
      其中 a=policy[s]
    """
    V_new = V.copy()
    for s in states:
        if s in terminal:
            V_new[s] = 0.0
            continue

        a = policy[s]
        # 确定性转移：这里只会有一个 s'
        v = 0.0
        for s2, p in P[(s, a)].items():
            r = R[(s, a, s2)]
            v += p * (r + gamma * V[s2])
        V_new[s] = v
    return V_new


def policy_evaluation(policy, gamma, states, terminal, actions, P, R, tol=1e-12, max_iter=10000):
    """
    已知模型下，对确定性策略做策略评估，求 V^π：
    - 初始化 V=0
    - 迭代 V <- T^π V 直到收敛
    """
    V = [0.0 for _ in states]
    for _ in range(max_iter):
        V_new = bellman_T_pi(V, policy, gamma, states, terminal, actions, P, R)
        delta = max(abs(a - b) for a, b in zip(V_new, V))
        V = V_new
        if delta < tol:
            break
    return V


def compute_Q_for_policy(policy, s, a, gamma, states, terminal, actions, P, R):
    """
    对固定策略 policy 计算 Q^π(s,a)（已知模型）：
    Q^π(s,a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V^π(s')]
    其中 V^π 是该策略下的状态价值。

    逻辑：
    - 先用策略评估算出 V^π
    - 再按上式计算单个 (s,a) 的 Q
    """
    V = policy_evaluation(policy, gamma, states, terminal, actions, P, R)

    # 终止态的 Q 视为0
    if s in terminal:
        return 0.0

    q = 0.0
    for s2, p in P[(s, a)].items():
        r = R[(s, a, s2)]
        q += p * (r + gamma * V[s2])
    return q


if __name__ == "__main__":
    gamma = 0.95
    states, terminal, actions, P, R = build_mdp_model()
    policies = enumerate_deterministic_policies()

    # 1) 评估所有策略得到 V^π
    V_list = []
    for pi in policies:
        V_list.append(policy_evaluation(pi, gamma, states, terminal, actions, P, R))

    # 2) 按定义取最大得到 V*
    V_star = [0.0 for _ in states]
    for s in states:
        V_star[s] = max(V[s] for V in V_list)

    # 3) 按定义枚举后续策略求 Q*(s,a)=max_π Q^π(s,a)
    Q_star = {(s, a): 0.0 for s in states for a in actions}
    for s in states:
        for a in actions:
            Q_star[(s, a)] = max(compute_Q_for_policy(pi, s, a, gamma, states, terminal, actions, P, R) for pi in policies)

    # 4) 验证 V*(s)=max_a Q*(s,a)，并给出贪心动作
    print(f"gamma={gamma}\n")
    print("State | V*(s)    | Q*(s,LEFT)  Q*(s,RIGHT) | max_a Q* | greedy action")
    print("------+----------+--------------------------+----------+--------------")
    for s in states:
        qL = Q_star[(s, LEFT)]
        qR = Q_star[(s, RIGHT)]
        max_q = max(qL, qR)
        greedy = "RIGHT" if qR >= qL else "LEFT"
        print(f"{s:>5} | {V_star[s]:>8.6f} | {qL:>10.6f}  {qR:>10.6f} | {max_q:>8.6f} | {greedy}")

    # 5) 找出一个最优策略（在枚举中挑一个使 V^π(1) 最大的策略作为示例）
    #    注意：最优策略可能不唯一，这里只展示一个。
    best_idx = max(range(len(policies)), key=lambda i: V_list[i][1])
    best_policy = policies[best_idx]
    print("\nOne optimal deterministic policy (mapping s->a for s=1,2,3):", best_policy)
