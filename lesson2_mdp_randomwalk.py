import random
from dataclasses import dataclass

LEFT, RIGHT = 0, 1

@dataclass
class StepResult:
    next_state: int
    reward: float
    done: bool

class RandomWalkMDP:
    """
    状态: 0..6，其中 0 和 6 为终止状态
    动作: LEFT(0), RIGHT(1)
    奖励: 进入状态6时 reward=+1，否则0
    """
    def __init__(self):
        self.terminal_left = 0
        self.terminal_right = 6
        self.start_state = 3

    def reset(self) -> int:
        return self.start_state

    def step(self, state: int, action: int) -> StepResult:
        if state in (self.terminal_left, self.terminal_right):
            return StepResult(state, 0.0, True)

        next_state = state - 1 if action == LEFT else state + 1
        done = next_state in (self.terminal_left, self.terminal_right)
        reward = 1.0 if next_state == self.terminal_right else 0.0
        return StepResult(next_state, reward, done)

def random_policy(state: int) -> int:
    # π(a|s)：这里用最简单的均匀随机策略
    return random.choice([LEFT, RIGHT])

def generate_episode(env: RandomWalkMDP, policy, max_steps=100):
    """
    采样一条 episode:
    返回 (states, actions, rewards) 序列
    states: S0, S1, ...
    actions: A0, A1, ...
    rewards: R1, R2, ... (与 Sutton 记法一致)
    """
    states, actions, rewards = [], [], []

    s = env.reset()
    states.append(s)

    for _ in range(max_steps):
        a = policy(s)
        sr = env.step(s, a)
        actions.append(a)
        rewards.append(sr.reward)

        s = sr.next_state
        states.append(s)

        if sr.done:
            break

    return states, actions, rewards

def discounted_return(rewards, gamma: float) -> float:
    """
    计算从t=0开始的回报 G0 = r1 + gamma*r2 + ...
    """
    G = 0.0
    power = 1.0
    for r in rewards:
        G += power * r
        power *= gamma
    return G

if __name__ == "__main__":
    random.seed(42)

    env = RandomWalkMDP()
    gamma = 0.95

    states, actions, rewards = generate_episode(env, random_policy)
    G0 = discounted_return(rewards, gamma)

    print("=== One episode ===")
    print("States: ", states)
    print("Actions:", ["L" if a == LEFT else "R" for a in actions])
    print("Rewards:", rewards)
    print(f"gamma={gamma}, Return G0={G0:.4f}")

    # 额外：多采样几条，直观看“期望回报”的概念
    n = 2000
    total = 0.0
    for _ in range(n):
        _, _, rs = generate_episode(env, random_policy)
        total += discounted_return(rs, gamma)
    print(f"Average return over {n} episodes (random policy): {total/n:.4f}")
