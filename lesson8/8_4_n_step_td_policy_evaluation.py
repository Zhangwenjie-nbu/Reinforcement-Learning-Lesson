# -*- coding: utf-8 -*-
"""
lesson8_4_n_step_td_policy_evaluation.py

整体在干什么？
1) 构造黑箱环境：1D Random Walk（可打滑），固定均匀随机策略 π（on-policy）。
2) 实现 n-step TD 策略评估：
      V(s_t) <- V(s_t) + alpha * (G_t^(n) - V(s_t))
   其中：
      G_t^(n) = r_{t+1} + gamma r_{t+2} + ... + gamma^{n-1} r_{t+n} + gamma^n V(s_{t+n})
   如果在 n 步内终止，则 bootstrap 项消失。
3) 运行两种 n：
   - n=1（TD(0)）
   - n=3（3-step TD）
   输出若干 checkpoint 的 V 估计，帮助你观察差异。

你需要掌握：
- n-step 是 MC 与 TD(0) 的连续桥梁：n=1 是 TD(0)，n=episode_length 是 MC
- 实现关键是：用一个滑动窗口/缓冲区延迟 n 步再更新
- 若 episode 提前终止，要正确“flush”剩余的更新（bootstrap 项为0）
"""

import random
from collections import defaultdict, deque

LEFT, RIGHT = 0, 1


class SlipperyRandomWalk:
    """
    1D Random Walk（可打滑）：
    - 状态：0..(n_states-1)
    - 起点：start_state
    - 终止：0 与 n_states-1
    - 动作：LEFT/RIGHT
    - slip：以 slip_prob 概率动作反转
    - 奖励：到达右端终止态给 +1，否则 0
    """

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
    """
    均匀随机策略：π(LEFT|s)=π(RIGHT|s)=0.5
    """

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, s: int) -> int:
        return self.rng.choice([LEFT, RIGHT])


def n_step_td_policy_evaluation(env, policy, gamma: float, alpha: float, n: int, n_episodes: int, max_steps=200):
    """
    n-step TD 策略评估（on-policy）：
    - 用一个 buffer 存最近的状态与奖励：
      states_buf: 存 s_t
      rewards_buf: 存 r_{t+1}
    - 当累计到 n 步后，就可以更新最早的那个状态 s_{t-n+1}：
      target = sum_{k=0}^{n-1} gamma^k r_{t-n+2+k} + gamma^n V(s_{t+1})
      注意：实现时更简单的方式是“每步入队，然后在满足长度时计算目标并弹出一次”

    终止时需要 flush：把 buffer 里剩余的状态用“无 bootstrap”形式依次更新完。
    """
    V = defaultdict(float)
    visit = defaultdict(int)

    # 终止态固定为0
    V[env.terminal_left] = 0.0
    V[env.terminal_right] = 0.0

    checkpoints = [10, 50, 200, 1000, n_episodes]

    for ep in range(1, n_episodes + 1):
        s = env.reset()

        states_buf = deque()   # 存放待更新的状态序列
        rewards_buf = deque()  # 存放对应的奖励序列（与 states_buf 对齐：reward 是从该 state 走出去得到的）

        # 先把初始状态入队（它的 reward 还未产生）
        states_buf.append(s)

        for step in range(max_steps):
            a = policy.act(s)
            s2, r, done = env.step(s, a)

            # 把这一步奖励入队（对应 states_buf 末尾那个 state 的 outgoing reward）
            rewards_buf.append(r)
            # 把下一状态入队
            states_buf.append(s2)

            # 当我们已经有 n 步奖励可用时（rewards_buf 长度 >= n），更新最早的 state
            if len(rewards_buf) >= n:
                s_update = states_buf[0]

                if not env.is_terminal(s_update):
                    # 计算 n-step target
                    G = 0.0
                    for k in range(n):
                        G += (gamma ** k) * rewards_buf[k]

                    # bootstrap state 是 states_buf[n]（因为 states_buf 长度比 rewards 多1）
                    s_boot = states_buf[n]
                    if env.is_terminal(s_boot):
                        v_boot = 0.0
                    else:
                        v_boot = V[s_boot]

                    target = G + (gamma ** n) * v_boot
                    V[s_update] += alpha * (target - V[s_update])
                    visit[s_update] += 1

                # 弹出最早的一步，窗口滑动
                states_buf.popleft()
                rewards_buf.popleft()

            s = s2
            if done:
                break

        # episode 结束后 flush：剩余 buffer 里每个 state 都要更新（此时 bootstrap 项为0，因为未来已结束）
        # 此时 rewards_buf 长度 = len(states_buf) - 1
        while len(rewards_buf) > 0:
            s_update = states_buf[0]

            if not env.is_terminal(s_update):
                G = 0.0
                for k in range(len(rewards_buf)):
                    G += (gamma ** k) * rewards_buf[k]
                target = G  # no bootstrap after termination
                V[s_update] += alpha * (target - V[s_update])
                visit[s_update] += 1

            states_buf.popleft()
            rewards_buf.popleft()

        if ep in checkpoints:
            snapshot = {s: round(V[s], 4) for s in range(env.n_states)}
            print(f"[n={n}] episode={ep:>5} | V_hat={snapshot}")

    return V, visit


if __name__ == "__main__":
    gamma = 0.95
    alpha = 0.1
    n_episodes = 5000

    env1 = SlipperyRandomWalk(n_states=7, start_state=3, slip_prob=0.2, seed=42)
    env3 = SlipperyRandomWalk(n_states=7, start_state=3, slip_prob=0.2, seed=42)  # 同 seed 便于对比
    policy = UniformRandomPolicy(seed=7)

    V_td0, _ = n_step_td_policy_evaluation(env1, policy, gamma, alpha, n=1, n_episodes=n_episodes)
    print()
    V_td3, _ = n_step_td_policy_evaluation(env3, policy, gamma, alpha, n=3, n_episodes=n_episodes)

    print("\nFinal comparison:")
    for s in range(env1.n_states):
        print(f"s={s}: TD0={V_td0[s]:.6f} | TD3={V_td3[s]:.6f}")
