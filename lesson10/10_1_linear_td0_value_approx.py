# -*- coding: utf-8 -*-
"""
lesson10_1_linear_td0_value_approx.py

整体在干什么？
1) 环境：1D Random Walk（可打滑），固定均匀随机策略 π（on-policy evaluation）。
2) 实现线性价值函数逼近：V_theta(s) = phi(s)^T theta
3) 使用线性 TD(0)（semi-gradient）更新参数：
      delta = r + gamma*V_theta(s') - V_theta(s)
      theta <- theta + alpha * delta * phi(s)
4) 对比两种特征：
   (A) one-hot 特征：相当于“用线性形式写的表格 TD”
   (B) 低维手工特征：用更少参数近似价值，观察泛化效果

你需要掌握：
- one-hot 特征时，线性模型就是表格模型的等价写法
- 低维特征时，一个参数会影响多个状态，实现泛化，但也会带来近似误差
"""

import random
from dataclasses import dataclass
from typing import List
import numpy as np

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


@dataclass
class LinearValueFunction:
    """
    线性价值函数：V_theta(s) = phi(s)^T theta
    """
    theta: np.ndarray

    def value(self, phi_s: np.ndarray) -> float:
        return float(phi_s @ self.theta)

    def update(self, phi_s: np.ndarray, delta: float, alpha: float):
        self.theta += alpha * delta * phi_s


def phi_one_hot(s: int, n_states: int) -> np.ndarray:
    """
    one-hot 特征：维度 = n_states
    """
    v = np.zeros(n_states, dtype=np.float32)
    v[s] = 1.0
    return v


def phi_handcrafted(s: int, n_states: int) -> np.ndarray:
    """
    低维手工特征（示例，维度=3）：
    - x：归一化位置（0..1）
    - x^2：提供非线性形状能力（仍是线性模型，只是特征非线性）
    - bias：常数项
    终止态仍会产生特征，但我们训练时不会更新终止态（保持它们价值为0的约束由数据自然实现）。
    """
    x = s / (n_states - 1)
    return np.array([x, x * x, 1.0], dtype=np.float32)


def linear_td0(env, policy, phi_fn, d: int, gamma=0.95, alpha=0.05, n_episodes=5000, max_steps=200, seed=0):
    """
    线性 TD(0) 策略评估：
    - 初始化 theta=0
    - 对每一步：
        delta = r + gamma*V(s') - V(s)
        theta <- theta + alpha*delta*phi(s)
    """
    rng = random.Random(seed)
    lvf = LinearValueFunction(theta=np.zeros(d, dtype=np.float32))

    checkpoints = [10, 50, 200, 1000, n_episodes]

    for ep in range(1, n_episodes + 1):
        s = env.reset()

        for _ in range(max_steps):
            a = policy.act(s)
            s2, r, done = env.step(s, a)

            if env.is_terminal(s):
                break

            phi_s = phi_fn(s, env.n_states)
            phi_s2 = phi_fn(s2, env.n_states)

            v_s = lvf.value(phi_s)
            v_s2 = 0.0 if env.is_terminal(s2) else lvf.value(phi_s2)

            delta = r + gamma * v_s2 - v_s
            lvf.update(phi_s, delta, alpha)

            s = s2
            if done:
                break

        if ep in checkpoints:
            V_hat = []
            for st in range(env.n_states):
                if env.is_terminal(st):
                    V_hat.append(0.0)
                else:
                    V_hat.append(round(lvf.value(phi_fn(st, env.n_states)), 4))
            print(f"episode={ep:>5} | theta={np.round(lvf.theta,4)} | V_hat={V_hat}")

    return lvf


if __name__ == "__main__":
    gamma = 0.95
    n_episodes = 5000

    env1 = SlipperyRandomWalk(n_states=7, start_state=3, slip_prob=0.2, seed=42)
    env2 = SlipperyRandomWalk(n_states=7, start_state=3, slip_prob=0.2, seed=42)
    policy = UniformRandomPolicy(seed=7)

    print("=== (A) one-hot features (equivalent to tabular TD(0)) ===")
    lvf_onehot = linear_td0(
        env1, policy,
        phi_fn=phi_one_hot,
        d=env1.n_states,
        gamma=gamma, alpha=0.1, n_episodes=n_episodes, seed=0
    )

    print("\n=== (B) handcrafted low-dim features (d=3, generalization) ===")
    lvf_hand = linear_td0(
        env2, policy,
        phi_fn=phi_handcrafted,
        d=3,
        gamma=gamma, alpha=0.05, n_episodes=n_episodes, seed=0
    )

    print("\nFinal value estimates:")
    for s in range(env1.n_states):
        if env1.is_terminal(s):
            v1 = 0.0
            v2 = 0.0
        else:
            v1 = lvf_onehot.value(phi_one_hot(s, env1.n_states))
            v2 = lvf_hand.value(phi_handcrafted(s, env1.n_states))
        print(f"s={s}: V_onehot={v1:.4f} | V_handcrafted={v2:.4f}")
