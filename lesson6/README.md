# 第 6 课（6.1）：贝尔曼期望方程（Bellman Expectation Equation）——从定义推导到可计算结构

本节只讲一个点：**推导 (V^\pi(s)) 的贝尔曼期望方程**，并把“每一步在做什么、为什么能这么做”讲透。我们只做 (V^\pi) 的版本；(Q^\pi) 的贝尔曼方程放到 6.2。

1. 从定义出发完整推导：
   [
   V^\pi(s)=\sum_a\pi(a|s)\sum_{s'}P(s'|s,a)\left(R(s,a,s')+\gamma V^\pi(s')\right)
   ]
2. 明确推导中使用的关键工具：

   * 回报递推：(G_t=R_{t+1}+\gamma G_{t+1})
   * 条件期望的线性性（期望可拆）
   * 全概率公式 / 先对动作取期望，再对下一状态取期望
   * 马尔可夫性（未来只依赖 (S_t,A_t)）
3. 理解这条方程在说什么：**当前状态的价值 = 按策略加权的“一步奖励 + 折扣后继状态价值”**。

---

### Step 0：写清定义（不省略）

价值定义：
[
V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t=s]
]

回报递推（上一章已证明是定义等价变形）：
[
G_t = R_{t+1} + \gamma G_{t+1}
]

---

### Step 1：把递推代入价值定义（在做什么？把长期量拆成一步+未来）

[
V^\pi(s) = \mathbb{E}*\pi[R*{t+1} + \gamma G_{t+1} \mid S_t=s]
]

**为什么能这么做？**
因为 (V^\pi) 就是 (G_t) 的条件期望，而 (G_t) 与右侧完全相等（恒等式），代入不改变任何意义。

---

### Step 2：用期望的线性性拆开（在做什么？把“和”拆成“期望的和”）

[
V^\pi(s)=\mathbb{E}*\pi[R*{t+1}\mid S_t=s] ;+; \gamma,\mathbb{E}*\pi[G*{t+1}\mid S_t=s]
]

**为什么能这么做？**
期望算子是线性的：(\mathbb{E}[X+Y]=\mathbb{E}[X]+\mathbb{E}[Y])，常数可提出。

---

### Step 3：引入动作，按全概率对 (A_t) 分解（在做什么？把“策略导致的随机动作”显式化）

在给定 (S_t=s) 时，动作 (A_t) 按策略采样：(A_t\sim\pi(\cdot|s))。
对任何随机变量 (X)，有：
[
\mathbb{E}[X\mid S_t=s] = \sum_a P(A_t=a\mid S_t=s)\ \mathbb{E}[X\mid S_t=s,A_t=a]
]
这里 (P(A_t=a\mid S_t=s)=\pi(a|s))。

所以：
[
V^\pi(s) = \sum_a \pi(a|s)\ \mathbb{E}!\left[R_{t+1}+\gamma G_{t+1}\mid S_t=s, A_t=a\right]
]

**为什么这一步关键？**
因为它把“策略”放进了表达式：后面才能看到 (\pi(a|s)) 的加权求和结构。

---

### Step 4：对下一状态按转移核分解（在做什么？把环境随机性显式化）

在给定 ((S_t=s,A_t=a)) 时，下一状态 (S_{t+1}\sim P(\cdot|s,a))。
再次使用全概率：
[
\mathbb{E}[Y\mid s,a] = \sum_{s'} P(s'|s,a)\ \mathbb{E}[Y\mid s,a,s']
]

令 (Y = R_{t+1}+\gamma G_{t+1})，得到：
[
V^\pi(s)=\sum_a\pi(a|s)\sum_{s'}P(s'|s,a)\ \mathbb{E}!\left[R_{t+1}+\gamma G_{t+1}\mid s,a,s'\right]
]

---

### Step 5：把“奖励机制”与“后继价值”识别出来（在做什么？把抽象期望变成可复用对象）

* 如果奖励是确定性函数 (R(s,a,s'))，那么：
  [
  \mathbb{E}[R_{t+1}\mid s,a,s']=R(s,a,s')
  ]
  若奖励还带噪声，则这里是期望奖励 (r(s,a,s'))，但形式不变。

* 对 (G_{t+1})：
  给定 (S_{t+1}=s')，并且从 (t+1) 起按同一策略 (\pi) 行为，
  [
  \mathbb{E}[G_{t+1}\mid S_{t+1}=s'] = V^\pi(s')
  ]
  这里依赖马尔可夫性：未来只需要知道当前 (s')，不需要更早历史。

因此：
[
\mathbb{E}[R_{t+1}+\gamma G_{t+1}\mid s,a,s']
= R(s,a,s') + \gamma V^\pi(s')
]

代回即可得到贝尔曼期望方程：
[
\boxed{
V^\pi(s)=\sum_a\pi(a|s)\sum_{s'}P(s'|s,a)\left(R(s,a,s')+\gamma V^\pi(s')\right)
}
]

---

### 这条方程到底在说什么（语义总结）

它说：在状态 (s) 的价值等于：

1. 你按策略 (\pi) 选择动作 (a)（策略加权）
2. 环境按 (P(s'|s,a)) 转移到 (s')（转移加权）
3. 得到一步奖励 (R(s,a,s'))
4. 加上后继状态价值的折扣 (\gamma V^\pi(s'))

也就是“当前价值 = 一步展望（one-step lookahead）的期望”。


# 第 6 课（6.2）：(Q^\pi(s,a)) 的贝尔曼期望方程——从定义推导到可计算迭代

本节只讲一个点：**推导并实现 (Q^\pi(s,a)) 的贝尔曼期望方程**，并用代码做策略评估（policy evaluation）迭代来数值求解 (Q^\pi)。最后用求得的 (Q^\pi) 验证：
[
V^\pi(s)=\sum_a \pi(a|s)Q^\pi(s,a)
]



1. 从定义严格推导：
   [
   Q^\pi(s,a)=\sum_{s'}P(s'|s,a)\left(R(s,a,s')+\gamma\sum_{a'}\pi(a'|s')Q^\pi(s',a')\right)
   ]
2. 明确推导依赖的关键点：

   * 回报递推 (G_t=R_{t+1}+\gamma G_{t+1})
   * 条件期望线性性
   * 马尔可夫性（从 (t+1) 起只需知道 (S_{t+1})）
3. 能用“语义”解释这条式子：**先走一步到 (s')，拿到奖励，再在 (s') 按策略继续**。

---

### Step 0：定义写清楚

[
Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s, A_t=a]
]

### Step 1：代入回报递推

[
Q^\pi(s,a)=\mathbb{E}*\pi[R*{t+1}+\gamma G_{t+1}\mid S_t=s, A_t=a]
]

### Step 2：对下一状态 (S_{t+1}) 做全概率分解

因为给定 ((s,a)) 后，(S_{t+1}\sim P(\cdot|s,a))：
[
Q^\pi(s,a)=\sum_{s'}P(s'|s,a)\ \mathbb{E}*\pi[R*{t+1}+\gamma G_{t+1}\mid s,a,s']
]

### Step 3：识别奖励项与后继项

* 奖励项：若奖励为确定性函数，则
  [
  \mathbb{E}[R_{t+1}\mid s,a,s']=R(s,a,s')
  ]
* 后继项：给定 (S_{t+1}=s')，从 (t+1) 起按策略继续，则
  [
  \mathbb{E}*\pi[G*{t+1}\mid S_{t+1}=s'] = V^\pi(s')
  ]
  又因为上一章讲过
  [
  V^\pi(s')=\sum_{a'}\pi(a'|s')Q^\pi(s',a')
  ]
  所以得到：
  [
  \boxed{
  Q^\pi(s,a)=\sum_{s'}P(s'|s,a)\left(R(s,a,s')+\gamma\sum_{a'}\pi(a'|s')Q^\pi(s',a')\right)
  }
  ]

---


# 第 6 课（6.3）：为什么贝尔曼备份迭代会收敛——“收缩映射”直觉与最小验证

本节只讲一个点：**贝尔曼期望算子为什么会把任意初值迭代到同一个固定点（即 (V^\pi) 或 (Q^\pi)）**。不做冗长数学证明，只把“最小必要结构”讲透：**(\gamma<1) 导致收缩（contraction）→ 固定点唯一 → 迭代收敛**。


1. 理解“算子迭代”的视角：(V_{k+1} = \mathcal{T}^\pi V_k)。
2. 知道什么是“固定点”（fixed point）：(V=\mathcal{T}^\pi V)。
3. 在直觉与公式层面理解“收缩”：
   [
   |\mathcal{T}^\pi V - \mathcal{T}^\pi U|*\infty \le \gamma |V-U|*\infty
   ]
4. 明白为什么这三件事连起来就得到：**唯一解 + 从任意初值收敛**。

---

### 1) 把贝尔曼方程写成“算子固定点”

6.1 我们得到贝尔曼期望方程（离散）：
[
V^\pi(s)=\sum_a\pi(a|s)\sum_{s'}P(s'|s,a)\left(R(s,a,s')+\gamma V^\pi(s')\right)
]
定义一个算子（把右边视作一个函数变换）：
[
(\mathcal{T}^\pi V)(s);=;\sum_a\pi(a|s)\sum_{s'}P(s'|s,a)\left(R(s,a,s')+\gamma V(s')\right)
]
那么贝尔曼方程就是：
[
V^\pi = \mathcal{T}^\pi V^\pi
]
也就是：**(V^\pi) 是算子 (\mathcal{T}^\pi) 的固定点**。

策略评估迭代本质上就是：
[
V_{k+1}=\mathcal{T}^\pi V_k
]

---

### 2) 关键不等式：(\mathcal{T}^\pi) 在 (|\cdot|_\infty) 下是 (\gamma)-收缩

我们看两个任意价值函数 (V,U)。对任意状态 (s)：

[
\begin{aligned}
(\mathcal{T}^\pi V)(s)-(\mathcal{T}^\pi U)(s)
&=\sum_a\pi(a|s)\sum_{s'}P(s'|s,a)\left(\gamma V(s')-\gamma U(s')\right)\
&=\gamma\sum_a\pi(a|s)\sum_{s'}P(s'|s,a)\left(V(s')-U(s')\right)
\end{aligned}
]

取绝对值并上界（这里用到“加权平均的绝对值不超过最大值”）：

[
\left|(\mathcal{T}^\pi V)(s)-(\mathcal{T}^\pi U)(s)\right|
\le \gamma \max_{s'}|V(s')-U(s')|
]

对所有 (s) 取最大值（这就是 (|\cdot|_\infty) 范数）：

[
|\mathcal{T}^\pi V-\mathcal{T}^\pi U|*\infty \le \gamma |V-U|*\infty
]

这句话的含义非常具体：**你输入的两个函数差多少，输出最多缩小到原来的 (\gamma) 倍**。
只要 (\gamma<1)，差距会被不断压缩。

---

### 3) 为什么“收缩”意味着“唯一解 + 迭代收敛”

这背后是一个经典结论（Banach 不动点定理的直觉版）：

* 如果一个算子是收缩（系数 < 1），它就**只有一个固定点**。
* 从任意初值开始反复应用该算子，序列都会**收敛到这个唯一固定点**。
* 收敛速度是几何级数：误差大约按 (\gamma^k) 衰减（因此 (\gamma) 越接近 1 越慢）。

这就是为什么你在 6.1/6.2 里看见 `delta` 会越来越小并收敛。

---
