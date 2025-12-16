import random
from collections import Counter

LEFT, RIGHT = 0, 1

class SlipperyChainMDP:
    """
    状态: 0..4
    动作: LEFT, RIGHT
    转移: 以 slip_prob 概率“打滑”到相反方向
    这里我们只关注 S_{t+1} 的分布，不引入复杂奖励
    """
    def __init__(self, slip_prob=0.2):
        self.slip_prob = slip_prob
        self.n_states = 5

    def step(self, s, a):
        # 打滑：动作反转
        if random.random() < self.slip_prob:
            a = LEFT if a == RIGHT else RIGHT

        if a == LEFT:
            s2 = max(0, s - 1)
        else:
            s2 = min(self.n_states - 1, s + 1)
        return s2

def estimate_transition(env, s, a, n=20000, seed=0):
    random.seed(seed)
    cnt = Counter()
    for _ in range(n):
        s2 = env.step(s, a)
        cnt[s2] += 1
    # 概率估计
    probs = {sp: cnt[sp] / n for sp in range(env.n_states)}
    return probs

if __name__ == "__main__":
    env = SlipperyChainMDP(slip_prob=0.2)

    # 我们只固定“当前状态 s=2、当前动作 a=RIGHT”
    # 然后用不同 seed 估计 P(s'|s=2,a=RIGHT) 的分布
    s = 2
    a = RIGHT

    p1 = estimate_transition(env, s, a, n=30000, seed=1)
    p2 = estimate_transition(env, s, a, n=30000, seed=999)

    print("Estimate of P(s'|s=2,a=RIGHT) with seed=1:   ", p1)
    print("Estimate of P(s'|s=2,a=RIGHT) with seed=999: ", p2)

    # 你也可以把“历史”想象成不同路径来到 s=2：
    # 但只要当前是 s=2、这一步选 RIGHT，下一步分布应一致（统计意义上）。
