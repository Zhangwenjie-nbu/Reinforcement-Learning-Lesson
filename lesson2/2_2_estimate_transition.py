import random
from collections import defaultdict

LEFT, RIGHT = 0, 1

class SlipperyChainMDP:
    """
    状态: 0..(n_states-1)
    动作: LEFT, RIGHT
    转移: 以 slip_prob 概率动作反转；边界处会夹紧（clamp）
    """
    def __init__(self, n_states=5, slip_prob=0.2):
        self.n_states = n_states
        self.slip_prob = slip_prob

    def step(self, s, a):
        # slip: flip action
        if random.random() < self.slip_prob:
            a = LEFT if a == RIGHT else RIGHT

        if a == LEFT:
            s2 = max(0, s - 1)
        else:
            s2 = min(self.n_states - 1, s + 1)
        return s2

def true_transition_probs(env: SlipperyChainMDP, s: int, a: int):
    """
    给出该环境的真实 P(s'|s,a)，用于对照。
    注意边界夹紧会导致“留在原地”的概率出现。
    """
    n = env.n_states
    slip = env.slip_prob

    # intended move
    if a == LEFT:
        s_intended = max(0, s - 1)
        s_slip = min(n - 1, s + 1)
    else:
        s_intended = min(n - 1, s + 1)
        s_slip = max(0, s - 1)

    probs = {sp: 0.0 for sp in range(n)}
    probs[s_intended] += (1.0 - slip)
    probs[s_slip] += slip
    return probs

def collect_data(env, n_steps=5000, seed=0, explore_policy="uniform"):
    """
    采样收集 (s,a,s') 数据。
    为了让所有(s,a)都有覆盖，这里用“均匀随机动作”，并且状态也均匀随机起点。
    """
    random.seed(seed)
    transitions = []  # list of (s,a,s2)

    for _ in range(n_steps):
        s = random.randrange(env.n_states)  # 随机挑一个当前状态，确保覆盖
        a = random.choice([LEFT, RIGHT])    # 均匀随机动作
        s2 = env.step(s, a)
        transitions.append((s, a, s2))

    return transitions

def estimate_P(env, transitions):
    """
    经验估计 \hat P(s'|s,a) = N(s,a,s') / N(s,a)
    返回：P_hat[(s,a)][s']
    """
    count_sa = defaultdict(int)
    count_sas2 = defaultdict(int)

    for s, a, s2 in transitions:
        count_sa[(s, a)] += 1
        count_sas2[(s, a, s2)] += 1

    P_hat = {}
    for s in range(env.n_states):
        for a in [LEFT, RIGHT]:
            denom = count_sa[(s, a)]
            dist = {sp: 0.0 for sp in range(env.n_states)}
            if denom > 0:
                for sp in range(env.n_states):
                    dist[sp] = count_sas2[(s, a, sp)] / denom
            P_hat[(s, a)] = dist

    return P_hat, count_sa

def l1_error(dist_true, dist_hat):
    return sum(abs(dist_true[sp] - dist_hat[sp]) for sp in dist_true.keys())

def run_once(n_steps, seed):
    env = SlipperyChainMDP(n_states=5, slip_prob=0.2)
    transitions = collect_data(env, n_steps=n_steps, seed=seed)
    P_hat, count_sa = estimate_P(env, transitions)

    # 计算每个(s,a)的L1误差，并汇总
    errs = []
    for s in range(env.n_states):
        for a in [LEFT, RIGHT]:
            pt = true_transition_probs(env, s, a)
            ph = P_hat[(s, a)]
            errs.append(l1_error(pt, ph))

    avg_err = sum(errs) / len(errs)
    min_coverage = min(count_sa[(s,a)] for s in range(env.n_states) for a in [LEFT, RIGHT])
    return avg_err, min_coverage

if __name__ == "__main__":
    # 看看样本数增加时，误差如何下降
    for n_steps in [200, 500, 2000, 10000, 50000]:
        avg_err, min_cov = run_once(n_steps=n_steps, seed=42)
        print(f"n_steps={n_steps:>6} | min N(s,a)={min_cov:>5} | avg L1 error={avg_err:.4f}")
