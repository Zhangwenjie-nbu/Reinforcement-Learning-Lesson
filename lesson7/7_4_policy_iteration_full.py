# -*- coding: utf-8 -*-
"""
lesson7_4_policy_iteration_full.py

整体在干什么？
1) 构造一个小离散 MDP（状态0..4，终止态0和4，动作LEFT/RIGHT），模型(P,R)已知。
2) 实现 Policy Iteration（策略迭代）：
   - Policy Evaluation：用贝尔曼期望方程迭代求 V^π
   - Policy Improvement：用 Q^π 贪心更新策略 π'，其中
         Q^π(s,a)=Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ V^π(s') ]
3) 循环执行评估与改进，直到策略不再变化（达到稳定策略）。
4) 打印每轮的策略与价值，观察其在有限步内收敛到最优策略。

你需要掌握：
- PI 的“优化策略”不是靠梯度，而是靠：评估（期望）+ 贪心改进（argmax）
- 单调改进 + 策略集合有限 => 有限步收敛
- 收敛时策略对自己的 Q 已经贪心，因此满足最优性条件
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


def bellman_expectation_backup_V(V, s, policy_det, gamma, terminal, P, R):
    """
    对确定性策略 policy_det 做单个状态的贝尔曼期望备份：
      V_new(s) = Σ_{s'} P(s'|s,a) [ R + γ V(s') ]
    其中 a = policy_det[s]

    终止态返回0。
    """
    if s in terminal:
        return 0.0

    a = policy_det[s]
    v_new = 0.0
    for s2, p in P[(s, a)].items():
        r = R[(s, a, s2)]
        v_new += p * (r + gamma * V[s2])
    return v_new


def policy_evaluation(policy_det, gamma, states, terminal, P, R, tol=1e-12, max_iter=10000):
    """
    Policy Evaluation：已知模型下评估确定性策略，求 V^π。
    用迭代法：
      V <- T^π V
    直到收敛。
    """
    V = [0.0 for _ in states]
    for _ in range(max_iter):
        V_new = V.copy()
        delta = 0.0
        for s in states:
            V_new[s] = bellman_expectation_backup_V(V, s, policy_det, gamma, terminal, P, R)
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < tol:
            break
    return V


def compute_Q_from_V(V, gamma, states, terminal, actions, P, R):
    """
    已知模型下，由 V^π 计算 Q^π(s,a)：
      Q^π(s,a)=Σ_{s'} P(s'|s,a) [ R + γ V^π(s') ]
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


def policy_improvement(Q, states, terminal):
    """
    Policy Improvement：对 Q^π 贪心，得到确定性策略 π'。
    并列时偏向 RIGHT，仅为输出稳定。
    """
    policy_new = {}
    for s in states:
        if s in terminal:
            continue
        qL = Q[(s, LEFT)]
        qR = Q[(s, RIGHT)]
        policy_new[s] = RIGHT if qR >= qL else LEFT
    return policy_new


def format_policy(policy_det):
    """
    以更可读的方式格式化策略映射。
    """
    items = []
    for s in sorted(policy_det.keys()):
        items.append(f"{s}:{'R' if policy_det[s]==RIGHT else 'L'}")
    return "{" + ", ".join(items) + "}"


def policy_iteration(gamma=0.95, max_outer_iter=50):
    """
    完整 Policy Iteration：
    - 初始化一个策略（这里选一个明显次优：对 1,2,3 全选 LEFT）
    - 循环：评估 -> 改进
    - 直到策略稳定或达到最大轮数

    返回：
    - policy_det: 最终策略
    - V: 最终价值
    """
    states, terminal, actions, P, R = build_mdp_model()

    # 初始化策略：全向左（通常很差，便于观察改进）
    policy_det = {1: LEFT, 2: LEFT, 3: LEFT}

    for k in range(max_outer_iter):
        V = policy_evaluation(policy_det, gamma, states, terminal, P, R)
        Q = compute_Q_from_V(V, gamma, states, terminal, actions, P, R)
        policy_new = policy_improvement(Q, states, terminal)

        changed = (policy_new != policy_det)

        print(f"=== Outer iter {k} ===")
        print("policy:", format_policy(policy_det))
        print("V:     ", [f"{v:.6f}" for v in V])
        print("Q(1):  ", f"L={Q[(1,LEFT)]:.6f}", f"R={Q[(1,RIGHT)]:.6f}")
        print("Q(2):  ", f"L={Q[(2,LEFT)]:.6f}", f"R={Q[(2,RIGHT)]:.6f}")
        print("Q(3):  ", f"L={Q[(3,LEFT)]:.6f}", f"R={Q[(3,RIGHT)]:.6f}")
        print("improved ->", format_policy(policy_new), "| changed:", changed)
        print()

        policy_det = policy_new
        if not changed:
            break

    # 最终再评估一次，返回稳定策略的 V
    V = policy_evaluation(policy_det, gamma, states, terminal, P, R)
    return policy_det, V


if __name__ == "__main__":
    final_policy, final_V = policy_iteration(gamma=0.95, max_outer_iter=50)
    print("Final policy:", format_policy(final_policy))
    print("Final V:     ", [round(v, 6) for v in final_V])
