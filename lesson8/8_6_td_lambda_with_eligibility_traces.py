# -*- coding: utf-8 -*-
"""
lesson8_6_td_lambda_with_eligibility_traces.py

整体在干什么？
1) 构造黑箱环境：1D Random Walk（可打滑），固定均匀随机策略 π（on-policy）。
2) 实现 TD(0) 与 TD(λ) 两种策略评估（状态价值）：
   - TD(0)：V(S_t) <- V(S_t) + alpha * delta_t
   - TD(λ)：引入 eligibility trace e(s)
       e(s) <- gamma*lambda*e(s) + 1[s==S_t]
       V(s) <- V(s) + alpha*delta_t*e(s)   (对所有 s)
3) 训练若干 episode，并在 checkpoint 打印 V 的估计用于对比。

你需要掌握：
- eligibility trace 让“一步 TD 误差”能影响过去近期访问过的多个状态（信用分配）
- e(s) 的指数衰减系数是 gamma*lambda，对应 λ-return 的几何加权结构
- 这是表格版实现；深度版会用函数逼近与 trace 的向量/梯度形式（后面再讲）
"""

import random
from collections import defaultdict

LEFT, RIGHT = 0, 1


class SlipperyRandomWalk:
    def __init__(self, n_states=7, start_state=3, slip_prob=0.2, seed=0):
        self.n_states = n_states
        self.start_state = start_state
        self.slip_prob = slip_prob
        self.rng = random.Random(seed)
        self.terminal_left = 0
        self.terminal_right = n_states - 1

    def reset(self):
        return self.start_state

    def is_terminal(self, s: int) -> bool:
        return s == self.terminal_left or s == self.terminal_right

    def step(self, s: int, a: int):
        if self.is_terminal(s):
            return s, 0.0, True
        if self.rng.random() < self.slip_prob:
            a = LEFT if a == RIGHT else RIGHT
        s2 = max(self.terminal_left, s - 1) if a == LEFT else min(self.terminal_right, s + 1)
        done = self.is_terminal(s2)
        r = 1.0 if s2 == self.terminal_right else 0.0
        return s2, r, done


class UniformRandomPolicy:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, s: int) -> int:
        return self.rng.choice([LEFT, RIGHT])


def td0_episode(env, policy, V, gamma: float, alpha: float, max_steps=200):
    """
    跑一条 episode 的 TD(0) 更新：
    - 每一步计算 delta = r + gamma*V(s') - V(s)
    - 只更新当前状态 V(s) += alpha*delta
    """
    s = env.reset()
    for _ in range(max_steps):
        a = policy.act(s)
        s2, r, done = env.step(s, a)

        if not env.is_terminal(s):
            v_next = 0.0 if env.is_terminal(s2) else V[s2]
            delta = r + gamma * v_next - V[s]
            V[s] += alpha * delta

        s = s2
        if done:
            break


def td_lambda_episode(env, policy, V, gamma: float, alpha: float, lam: float, max_steps=200):
    """
    跑一条 episode 的 TD(λ)（accumulating traces）更新：
    - 维护 eligibility trace e(s)
    - 每步：
        delta = r + gamma*V(s') - V(s)
        e(s) <- gamma*lambda*e(s) + 1[s==S_t]
        对所有状态：V(x) <- V(x) + alpha*delta*e(x)
    """
    e = defaultdict(float)  # eligibility traces
    s = env.reset()

    for _ in range(max_steps):
        a = policy.act(s)
        s2, r, done = env.step(s, a)

        if env.is_terminal(s):
            break

        v_next = 0.0 if env.is_terminal(s2) else V[s2]
        delta = r + gamma * v_next - V[s]

        # 更新 traces：所有 trace 衰减
        for x in list(e.keys()):
            e[x] *= gamma * lam
            # 清理很小的 trace，避免字典越来越大
            if abs(e[x]) < 1e-12:
                del e[x]

        # 当前状态 trace +1
        e[s] += 1.0

        # 用 traces 分配 TD 误差给多个状态
        for x, ex in e.items():
            V[x] += alpha * delta * ex

        s = s2
        if done:
            break


def train_compare_td0_vs_tdlambda(gamma=0.95, alpha=0.1, lam=0.8, n_episodes=5000):
    """
    对比训练 TD(0) 与 TD(λ) 的估计。
    """
    env0 = SlipperyRandomWalk(n_states=7, start_state=3, slip_prob=0.2, seed=42)
    envL = SlipperyRandomWalk(n_states=7, start_state=3, slip_prob=0.2, seed=42)
    policy = UniformRandomPolicy(seed=7)

    V_td0 = defaultdict(float)
    V_tdl = defaultdict(float)

    # 终止态固定为0
    for V in (V_td0, V_tdl):
        V[env0.terminal_left] = 0.0
        V[env0.terminal_right] = 0.0

    checkpoints = [10, 50, 200, 1000, n_episodes]

    for ep in range(1, n_episodes + 1):
        td0_episode(env0, policy, V_td0, gamma, alpha)
        td_lambda_episode(envL, policy, V_tdl, gamma, alpha, lam)

        if ep in checkpoints:
            snap0 = {s: round(V_td0[s], 4) for s in range(env0.n_states)}
            snapL = {s: round(V_tdl[s], 4) for s in range(env0.n_states)}
            print(f"episode={ep:>5} | TD0={snap0}")
            print(f"           | TDλ={snapL}\n")

    return V_td0, V_tdl


if __name__ == "__main__":
    gamma = 0.95
    alpha = 0.1
    lam = 0.8
    n_episodes = 5000

    V0, VL = train_compare_td0_vs_tdlambda(gamma=gamma, alpha=alpha, lam=lam, n_episodes=n_episodes)

    print("Final comparison:")
    for s in range(7):
        print(f"s={s}: TD0={V0[s]:.6f} | TDλ={VL[s]:.6f}")
