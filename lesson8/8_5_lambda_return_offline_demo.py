# -*- coding: utf-8 -*-
"""
lesson8_5_lambda_return_offline_demo.py

整体在干什么？
1) 构造黑箱环境：1D Random Walk（可打滑），固定均匀随机策略 π。
2) 采样一条 episode，得到 states=[S0..ST], rewards=[R1..RT]。
3) 给定一个当前价值函数估计 V（这里用0初始化，仅用于演示 bootstrap 项）。
4) 对每个时间步 t：
   - 计算多个 n-step return: G_t^(n)
   - 用几何权重 w_n=(1-lambda)*lambda^(n-1) 混合，得到 λ-return: G_t^λ
5) 打印每个 t 的若干 n-step return 与 λ-return，让你看到“加权混合”的具体数值形态。

你需要掌握：
- λ-return 是对所有 n-step return 的几何加权平均
- 这是定义层的离线计算；在线实现需要 eligibility traces（下一节）
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


def generate_episode(env, policy, max_steps=200):
    """
    采样一条 episode：
    返回：
    - states: [S0, S1, ..., ST]  长度T+1
    - rewards: [R1, R2, ..., RT] 长度T
    """
    s = env.reset()
    states = [s]
    rewards = []

    for _ in range(max_steps):
        a = policy.act(s)
        s2, r, done = env.step(s, a)
        states.append(s2)
        rewards.append(r)
        s = s2
        if done:
            break
    return states, rewards


def n_step_return(states, rewards, V, t: int, n: int, gamma: float, terminal_states: set):
    """
    计算一条 episode 中，从时间步 t 开始的 n-step return: G_t^(n)

    输入：
    - states: [S0..ST]
    - rewards: [R1..RT]
    - t: 时间步索引（对应状态 S_t）
    - n: n-step
    规则：
    - 累计前 n 步真实奖励（若提前终止，则只累积到终止）
    - 如果存在 S_{t+n} 且非终止，则加 bootstrap: gamma^n * V(S_{t+n})
      否则 bootstrap 项为0
    """
    T = len(rewards)  # episode length
    G = 0.0

    # 累积真实奖励
    for k in range(n):
        idx = t + k
        if idx >= T:
            break
        G += (gamma ** k) * rewards[idx]

        # 若到达终止态，后续奖励为0并停止
        if states[idx + 1] in terminal_states:
            return G

    # bootstrap
    t_n = t + n
    if t_n <= T:
        s_boot = states[t_n]
        if s_boot in terminal_states:
            return G
        else:
            return G + (gamma ** n) * V[s_boot]

    return G


def lambda_return(states, rewards, V, t: int, gamma: float, lam: float, terminal_states: set):
    """
    离线计算 λ-return:
      G_t^λ = (1-λ) Σ_{n=1..∞} λ^{n-1} G_t^(n)

    对 episodic 任务，n 不必到无穷，最多到剩余步数即可（再大也相同）。
    """
    T = len(rewards)
    max_n = T - t  # 最多还能走多少步奖励
    if max_n <= 0:
        return 0.0

    G_lam = 0.0
    for n in range(1, max_n + 1):
        w = (1.0 - lam) * (lam ** (n - 1))
        G_n = n_step_return(states, rewards, V, t, n, gamma, terminal_states)
        G_lam += w * G_n

    return G_lam


if __name__ == "__main__":
    gamma = 0.95
    lam = 0.8

    env = SlipperyRandomWalk(n_states=7, start_state=3, slip_prob=0.2, seed=42)
    policy = UniformRandomPolicy(seed=7)

    states, rewards = generate_episode(env, policy)
    terminal_states = {env.terminal_left, env.terminal_right}

    # 用0初始化 V（演示 bootstrap；更真实的做法是用已有 TD0/MC 得到的 V 估计）
    V = defaultdict(float)
    V[env.terminal_left] = 0.0
    V[env.terminal_right] = 0.0

    print("Episode states:", states)
    print("Episode rewards:", rewards)
    print(f"\ngamma={gamma}, lambda={lam}\n")

    # 对每个 t 打印若干 n-step 与 lambda-return
    T = len(rewards)
    for t in range(T):
        s = states[t]
        # 打印 n=1,2,3 与 lambda-return
        g1 = n_step_return(states, rewards, V, t, 1, gamma, terminal_states)
        g2 = n_step_return(states, rewards, V, t, 2, gamma, terminal_states)
        g3 = n_step_return(states, rewards, V, t, 3, gamma, terminal_states)
        glam = lambda_return(states, rewards, V, t, gamma, lam, terminal_states)
        print(f"t={t:>2}, s={s}: G1={g1:.4f}, G2={g2:.4f}, G3={g3:.4f}, G^lambda={glam:.4f}")
