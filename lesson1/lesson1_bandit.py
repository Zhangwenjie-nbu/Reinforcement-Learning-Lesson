import random

class KArmedBandit:
    """
    K臂老虎机环境：每个动作a有一个固定但未知的奖励均值 mu[a]
    每次pull(a)返回带噪声的奖励（这里用简单的伯努利/高斯都可以）
    为了尽量零依赖，这里用伯努利奖励：reward ∈ {0,1}
    """
    def __init__(self, mus):
        self.mus = mus  # list[float], 每个动作获奖概率

    def pull(self, a: int) -> float:
        p = self.mus[a]
        return 1.0 if random.random() < p else 0.0


class EpsilonGreedyAgent:
    """
    ε-greedy 智能体：维护动作价值估计Q[a]和选择次数N[a]
    """
    def __init__(self, k: int, epsilon: float):
        self.k = k
        self.epsilon = epsilon
        self.Q = [0.0] * k
        self.N = [0] * k

    def act(self) -> int:
        # 探索：随机动作
        if random.random() < self.epsilon:
            return random.randrange(self.k)
        # 利用：选择当前Q最大的动作（若并列，随机挑一个，避免偏置）
        max_q = max(self.Q)
        candidates = [i for i, q in enumerate(self.Q) if q == max_q]
        return random.choice(candidates)

    def update(self, a: int, r: float):
        # 在线样本均值更新：Q <- Q + (1/N)*(r - Q)
        self.N[a] += 1
        alpha = 1.0 / self.N[a]
        self.Q[a] += alpha * (r - self.Q[a])


def train(mus, epsilon=0.1, steps=5000, seed=0):
    random.seed(seed)

    env = KArmedBandit(mus)
    agent = EpsilonGreedyAgent(k=len(mus), epsilon=epsilon)

    total_reward = 0.0
    for t in range(steps):
        a = agent.act()
        r = env.pull(a)
        agent.update(a, r)
        total_reward += r

    return agent, total_reward


if __name__ == "__main__":
    # 真实但未知的各臂获奖概率（你可以改）
    mus = [0.05, 0.15, 0.12, 0.30, 0.25]

    agent, total_reward = train(mus, epsilon=0.1, steps=8000, seed=42)

    print("=== Results ===")
    print(f"True mus:      {mus}")
    print(f"Estimated Q:   {[round(q, 4) for q in agent.Q]}")
    print(f"Action counts: {agent.N}")
    best_true = max(range(len(mus)), key=lambda i: mus[i])
    best_est  = max(range(len(agent.Q)), key=lambda i: agent.Q[i])
    print(f"Best true arm: {best_true}, mu={mus[best_true]}")
    print(f"Best est  arm: {best_est}, Q={agent.Q[best_est]:.4f}")
    print(f"Total reward:  {total_reward} / 8000 = {total_reward/8000:.4f}")
