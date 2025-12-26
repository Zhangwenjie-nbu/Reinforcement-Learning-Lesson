# 第 8 课（8.1）：Monte Carlo（MC）策略评估——在不知道模型 (P,R) 时如何估计 (V^\pi)

到目前为止，我们的 DP（策略评估/迭代）都默认你知道 (P(s'|s,a)) 和 (R(s,a,s'))。现实中通常不知道模型，只能与环境交互拿到样本轨迹。本节只讲一个点：**用 Monte Carlo 采样回报来估计 (V^\pi(s))**，并讲清楚它“为什么对、哪里慢”。

---

1. 用一句话描述 MC 评估：**对固定策略 (\pi)，把每次从状态出发得到的回报 (G) 当样本，用样本均值估计 (V^\pi(s)=\mathbb{E}[G|S=s])**。
2. 区分 MC 的两种常见版本：First-Visit vs Every-Visit。
3. 理解 MC 的优点：不需要模型、估计“无偏”（在充分采样下）。
4. 理解 MC 的局限：方差大、通常需要 episode 结束（对持续任务不友好）。

---

### 1) MC 的核心公式（把期望换成样本均值）

定义回顾：
[
V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]
]

MC 的做法是：反复采样 episode，在每次 episode 中收集“从某个状态开始的回报样本”。
当我们拿到 (N) 个从状态 (s) 产生的回报样本 (G^{(1)},\dots,G^{(N)}) 时：
[
\hat V^\pi(s)=\frac{1}{N}\sum_{i=1}^N G^{(i)}
]
这就是统计学里最朴素的“用样本均值估计期望”。

---

### 2) First-Visit vs Every-Visit（本节只讲定义差别）

在一条 episode 里，某个状态 (s) 可能出现多次。

* **First-Visit MC**：只用该 episode 中第一次访问 (s) 时的回报样本更新 (V(s))。
* **Every-Visit MC**：该 episode 中每次访问 (s) 都用对应回报样本更新 (V(s))。

二者在无穷采样下都能收敛到 (V^\pi)，但样本利用方式与方差会不同。

---

### 3) 为什么 MC “正确”（直觉）

MC 没做任何近似递推，它直接用 (G) 的样本均值逼近期望值。只要：

* 你一直按同一个策略 (\pi) 产生数据（on-policy）
* 状态 (s) 被访问到的次数足够多

那么根据大数定律，样本均值会收敛到真实期望：(\hat V^\pi(s)\to V^\pi(s))。

---

### 4) MC 的主要局限

* **方差大**：因为 (G) 是长回报和，随机性累计导致波动大。
* **通常要等 episode 结束才能计算回报**：对持续任务需要额外处理（截断、平均回报、或改用 TD）。
  这也是为什么 TD 学习会在 MC 之后登场（8.2 起）。

---


# 第 8 课（8.2）：TD(0) 策略评估——不用等 episode 结束，如何在线估计 (V^\pi)

上一节 MC 的核心问题是：**必须等一整条 episode 结束才能得到回报 (G_t)**，而且方差往往很大。本节只讲一个点：**TD(0) 用“一步目标” (r+\gamma V(s')) 来替代完整回报 (G)，实现在线、低方差的策略评估**。

---

1. 写出 TD(0) 的更新式并解释每一项的含义：
   [
   V(s)\leftarrow V(s)+\alpha\Big(\underbrace{r+\gamma V(s')}*{\text{TD target}}-\underbrace{V(s)}*{\text{current}}\Big)
   ]
2. 理解 TD 误差（TD error）：
   [
   \delta = r+\gamma V(s')-V(s)
   ]
3. 明确 TD(0) 与 MC 的关键差别：

   * MC：目标是真实回报样本 (G_t)（无偏、方差大、要等结束）
   * TD：目标用当前估计 (V(s')) 自举（有偏、方差小、可在线）

---

### 1) TD(0) 的核心思想：用“一步展望”近似未来

我们从贝尔曼期望方程的语义来理解（不需要模型）：
[
V^\pi(s)=\mathbb{E}[R_{t+1}+\gamma V^\pi(S_{t+1})\mid S_t=s]
]
如果我们有一个当前的估计 (V(\cdot))，那么在一次样本转移 ((s,a,r,s')) 上，自然的“一步目标”就是：
[
\text{TD target}= r+\gamma V(s')
]
然后把 (V(s)) 向这个目标挪一点点（学习率 (\alpha)）。

---

### 2) TD 误差 (\delta) 在做什么

[
\delta = r+\gamma V(s')-V(s)
]

* (\delta>0)：说明你当前把 (V(s)) 估低了，应上调
* (\delta<0)：说明你估高了，应下调
  它是一个**局部一致性误差**：让你的估计更符合“贝尔曼一致性”。

---

### 3) TD(0) 为什么可以在线更新

因为每一步交互就能得到 ((r,s'))，并且目标只依赖当前估计 (V(s'))。不需要等未来整个回合结束。

---

### 4) 偏差-方差直觉（你只需要记住这句）

* MC：目标是真实回报样本 (G_t)，**无偏但高方差**
* TD：目标用 (V(s')) 代替未来回报，**有偏但低方差**，通常更快更稳定

---

# 第 8 课（8.3）：TD(0) 到底在逼近什么——它的“期望更新”为什么对应贝尔曼期望方程的固定点

本节只讲一个点：**TD(0) 虽然每次用的是样本目标 (r+\gamma V(s'))，但在“期望意义”下，它是在逼近贝尔曼期望方程的固定点 (V^\pi)**。这回答两个关键疑问：

* TD 为什么“理论上能学到正确的 (V^\pi)”？
* TD 的更新到底在优化什么一致性？

---
1. 用一句话说明 TD(0) 的本质：**让 (V(s)) 逐步满足 (V(s)\approx \mathbb{E}[r+\gamma V(s')|s])**。
2. 写出 TD(0) 的期望更新形式，并看出它的固定点就是 (V^\pi)。
3. 理解“样本更新（随机）”与“期望算子（确定）”之间的关系：样本只是对期望的随机近似。

---

### Step 1：从 TD(0) 的更新式出发

TD(0) 一步更新：
[
V_{t+1}(S_t)=V_t(S_t)+\alpha\Big(r_{t+1}+\gamma V_t(S_{t+1})-V_t(S_t)\Big)
]
定义 TD 误差：
[
\delta_t = r_{t+1}+\gamma V_t(S_{t+1})-V_t(S_t)
]

---

### Step 2：把“随机更新”变成“期望更新”（关键点）

对一个固定状态 (s)，思考：当我们处于 (S_t=s) 时，这一步更新在期望上会把 (V(s)) 往哪里推？

对条件 (S_t=s) 取期望：
[
\mathbb{E}[V_{t+1}(s)\mid S_t=s]
================================

V_t(s)+\alpha\Big(\mathbb{E}[r_{t+1}+\gamma V_t(S_{t+1})\mid S_t=s]-V_t(s)\Big)
]

注意：这里的期望只对“环境转移与策略采样导致的随机性”取。(V_t(\cdot)) 在这一刻是确定的函数。

---

### Step 3：识别出贝尔曼期望算子 (\mathcal{T}^\pi)

在固定策略 (\pi) 下，有贝尔曼期望算子：
[
(\mathcal{T}^\pi V)(s)=\mathbb{E}*\pi[r*{t+1}+\gamma V(S_{t+1})\mid S_t=s]
]

于是 TD 的期望更新就可以写成：
[
\mathbb{E}[V_{t+1}(s)\mid S_t=s]
================================

V_t(s)+\alpha\Big((\mathcal{T}^\pi V_t)(s)-V_t(s)\Big)
]

这句话非常关键：
**TD(0) 在期望意义下，是在把 (V_t) 往 (\mathcal{T}^\pi V_t) 拉近。**

---

### Step 4：为什么固定点就是 (V^\pi)

所谓固定点，就是更新不再改变值函数的点。看上面的式子，若对所有 (s)：
[
(\mathcal{T}^\pi V)(s)=V(s)
]
那么期望更新项为 0，系统在期望意义下保持不动。

而你在 6.1 已经学过：
[
V^\pi = \mathcal{T}^\pi V^\pi
]
因此 **(V^\pi) 就是 TD(0) 期望更新对应算子的固定点**。

直觉总结：

* MC 直接逼近 (\mathbb{E}[G])
* TD 逼近贝尔曼一致性 (V=\mathbb{E}[r+\gamma V'])
  二者最终指向同一个 (V^\pi)，只是逼近路径不同。

---
# 第 8 课（8.4）：n-step TD——在 MC 与 TD(0) 之间搭桥的“可调折中”

本节只讲一个点：**n-step TD 的目标到底是什么**。它把：

* TD(0) 的一步自举（低方差、有偏）
* MC 的完整回报（无偏、高方差）
  统一成一个可调参数 (n) 的家族：**用前 (n) 步真实奖励 + 最后一步 bootstrap**。

---

1. 写出 n-step return（n 步回报）定义：
   [
   G^{(n)}*t=\sum*{k=0}^{n-1}\gamma^k r_{t+k+1} + \gamma^n V(s_{t+n})
   ]
   若在 (t+n) 前 episode 终止，则 bootstrap 项消失（用实际终止回报）。
2. 看清它如何包含两个极端情况：

   * (n=1)：退化为 TD(0) 目标 (r_{t+1}+\gamma V(s_{t+1}))
   * (n\to\infty)（直到终止）：退化为 MC 回报 (G_t)
3. 理解偏差-方差折中方向：

   * (n) 越大，越接近 MC（偏差小、方差大、更新更延迟）
   * (n) 越小，越接近 TD(0)（偏差大、方差小、更新更在线）

---

### 1) n-step return 是怎么来的（直觉）

TD(0) 的问题是：未来全靠 (V(s')) 估计（自举偏差）。
MC 的问题是：未来全靠真实采样（高方差、要等终止）。
n-step 的想法是：
**先把未来的前 (n) 步“用真实奖励算出来”，剩下的更远未来再用 (V) 估计。**

因此目标写成：
[
G^{(n)}*t=\underbrace{r*{t+1}+\gamma r_{t+2}+\cdots+\gamma^{n-1}r_{t+n}}*{\text{前 n 步真实奖励}}
;+;\underbrace{\gamma^n V(s*{t+n})}_{\text{从第 n 步起 bootstrap}}
]

---

### 2) 终止情况如何处理（你必须掌握）

如果 episode 在 (t+n) 之前终止（比如在 (t+m) 终止，(m<n)），则没有 (s_{t+n})，bootstrap 项不存在，实际目标就是截断的 MC 回报：
[
G^{(n)}*t=\sum*{k=0}^{m-1}\gamma^k r_{t+k+1}
]
这点对实现非常关键，否则会访问越界。

---

### 3) 更新式（本质仍是“把当前估计往目标拉近”）

与 TD(0) 一样：
[
V(s_t)\leftarrow V(s_t)+\alpha\left(G^{(n)}_t - V(s_t)\right)
]
不同只是目标从“一步”变成“n 步”。

---
# 第 8 课（8.5）：TD(λ) 与 λ-return——把所有 n-step 目标“加权混合”成一个统一目标

本节只讲一个点：**λ-return 的定义**。它把 8.4 的 n-step return 家族用一个参数 (\lambda\in[0,1]) 进行几何加权，从而形成一个统一目标。下一节（8.6）再讲如何在线实现（eligibility traces）。

---
1. 写出 n-step return 与 λ-return 的关系：
   [
   G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}+\gamma^n V(s_{t+n})
   ]
   [
   G_t^\lambda=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}
   ]
2. 看清两个极端：

   * (\lambda=0\Rightarrow G_t^\lambda=G_t^{(1)})（TD(0)）
   * (\lambda\to 1\Rightarrow) 趋近 MC 回报（在 episodic 任务中直观成立）
3. 理解它在做的事：**把“短视、自举多”与“长视、真实回报多”用一个参数连续插值**。

---

### 1) 回忆：n-step 的目标是一族

n-step return：
[
G_t^{(n)}=\underbrace{r_{t+1}+\gamma r_{t+2}+\cdots+\gamma^{n-1}r_{t+n}}*{\text{前 n 步真实奖励}}
;+;\gamma^n V(s*{t+n})
]
它随 (n) 增大越来越像 MC。

---

### 2) λ-return：对所有 n-step 做几何加权平均

λ-return 定义：
[
\boxed{
G_t^\lambda=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}
}
]
权重是：
[
w_n=(1-\lambda)\lambda^{n-1}
]
这是一个几何分布形式，满足 (\sum_{n\ge1} w_n = 1)。含义：

* 小 (n) 的 return 权重大（更像 TD0）
* 大 (n) 的 return 仍有权重（更像 MC）
* (\lambda) 越大，大 (n) 权重衰减越慢，越“长视”

---

### 3) 两个极端为什么成立（你应该记住的直观）

* (\lambda=0)：只有 (n=1) 这一项权重非零
  [
  G_t^\lambda = G_t^{(1)} = r_{t+1}+\gamma V(s_{t+1})
  ]
  这就是 TD(0)。

* (\lambda\to 1)：几何权重更平坦，会更多纳入长 horizon 的 return，在 episodic 任务中会趋近 MC 回报（直觉：无限加权混合逐渐变成“更长的真实奖励链”）。

---

### 4) TD(λ) 的更新式（先给形式，不讲在线实现细节）

有了 (G_t^\lambda)，更新仍是：
[
V(s_t)\leftarrow V(s_t)+\alpha\left(G_t^\lambda - V(s_t)\right)
]
难点是：(G_t^\lambda) 看起来要用很多 n-step return 的组合，似乎不能在线高效算。
这就是下一节要解决的：**eligibility traces 让 TD(λ) 可以在线实现**。

---
# 第 8 课（8.6）：Eligibility Traces——TD(λ) 为什么能在线实现（把误差“分配给过去”）

上一节你看到 λ-return 的定义是“很多 n-step return 的加权混合”，看起来离线才能算。本节只讲一个点：**eligibility traces（资格迹）如何把“混合的多步目标”改写成“每一步一个 TD 误差 (\delta_t) + 一个随时间衰减的迹 (e)”的在线更新**。

---

## 微课 8.6（理论）：资格迹在做什么、为什么它等价于 λ-return（直觉版）

### 教学目标

学完你应能：

1. 写出 TD(λ)（状态价值版本）最经典的在线形式：

   * TD 误差：
     [
     \delta_t = r_{t+1} + \gamma V(S_{t+1}) - V(S_t)
     ]
   * 累积迹（accumulating trace）更新：
     [
     e(s)\leftarrow \gamma\lambda e(s) + \mathbf{1}[s=S_t]
     ]
   * 价值更新：
     [
     V(s)\leftarrow V(s) + \alpha,\delta_t, e(s)
     ]
2. 理解迹 (e(s)) 的语义：**“最近访问过的状态应该对当前 TD 误差负责多少”**。
3. 理解为什么它对应 λ-return 的混合：迹的衰减 ((\gamma\lambda)^k) 正好给过去 (k) 步的状态分配一个几何权重（与 λ-return 的几何混合一致）。

---

### 1) 为什么需要“把误差分配给过去”

TD(0) 每一步只更新当前状态 (S_t)。
但 n-step / λ-return 的思想是：当前的奖励信息应该影响最近若干步访问过的状态（信用分配）。资格迹就是一种在线的信用分配机制。

---

### 2) 迹 (e(s)) 的含义（你可以把它当作“最近访问强度”）

更新：
[
e(s)\leftarrow \gamma\lambda e(s) + \mathbf{1}[s=S_t]
]
解释：

* 如果某个状态最近刚访问过，它的 (e(s)) 会被加 1
* 随时间推进，如果不再访问它，(e(s)) 会按 (\gamma\lambda) 指数衰减
  所以 (e(s)) 大的状态，就是“最近经常访问过的状态”。

---

### 3) 为什么这样更新就实现了 λ-return（直觉，不做完整代数展开）

每一步你产生一个 TD 误差 (\delta_t)。你用 (e(s)) 把它分给很多状态：

[
V(s)\leftarrow V(s)+\alpha\delta_t e(s)
]

由于 (e(s)) 对“过去 k 步访问过的状态”的贡献大约是 ((\gamma\lambda)^k)，因此：

* 更近的状态得到更大权重
* 更远的状态得到更小权重（几何衰减）

这与 λ-return “对不同步长 n 做几何加权混合”的权重结构是一致的。结果就是：**你不用显式算所有 n-step return，却在更新效果上实现了相同的加权信用分配**。

---

## 微课 8.6（编码）：实现 TD(λ)（accumulating traces），观察其比 TD(0) 更“快地把奖励传回去”

我们继续使用同一个黑箱 random-walk 环境、固定随机策略。实现：

* TD(0)：只更新当前状态
* TD(λ)：维护迹 (e(s))，每步更新所有状态（迹不为 0 的状态）
  并对比输出估计值变化趋势。

为了保持实现最清晰，我们使用表格（dict）存 (V) 与 (e)。

---

## 代码：`lesson8_6_td_lambda_with_eligibility_traces.py`

```python
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
```

### 运行方式

```bash
python3 lesson8_6_td_lambda_with_eligibility_traces.py
```

### 你应该观察到什么（这一步在做什么，为什么这么做）

* TD(λ) 往往比 TD(0) 更快把“终点奖励”影响传回到更早出现的状态（尤其当奖励稀疏、episode 较长时更明显）。
* 这是因为 TD(λ) 每一步的 TD 误差会通过 trace 更新一串最近访问过的状态，而 TD(0) 只更新当前状态。

---

## 本节总结与下一节预告

### 已学内容（简要）

* λ-return 是 n-step return 的几何混合；eligibility traces 让这种混合可以在线实现。
* TD(λ) 的在线形式由三部分构成：TD 误差 (\delta_t)、迹更新 (e)、用迹分配误差更新 (V)。
* 你实现了表格版 TD(λ) 并与 TD(0) 对比，理解 trace 如何加速信用分配。

### 下一节目标（第 9 课 9.1）

我们进入“从评估到控制”的学习算法：
只讲一个点：**SARSA（on-policy TD control）**

* 它如何把 TD 更新从 (V) 扩展到 (Q)
* 更新目标为何是 (r+\gamma Q(s',a'))（而不是 max）

---

## 课程要求（提醒）

1. 慢进度、深讲解：每节只讲一个小点但讲透。
2. 每节包含理论微课 + 编码微课，代码用于验证本节核心点。
3. 代码注释规范：文件顶部整体说明；每个函数上方说明其功能与逻辑。
4. 每节末尾：回顾已学、预告下一节目标，并附“课程要求”摘要。
