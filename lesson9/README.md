# 第 9 课（9.1）：SARSA——从“评估”走向“控制”的第一步（on-policy TD control）

到目前为止你一直在做 **policy evaluation**：给定策略 (\pi)，学 (V^\pi)。本节只讲一个点：**如何在不知道模型的情况下，一边采样一边改进策略**。SARSA 是最经典的起点：它学习的是 **动作价值 (Q(s,a))**，并且是 **on-policy**（用你当前正在执行的策略产生的数据与目标一致）。

---

1. 明确 SARSA 学的对象：(Q^\pi(s,a))（并通过策略改进逐步变好）。
2. 写出 SARSA 的 TD 更新式并解释每一项：
   [
   Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\Big(r_{t+1}+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)\Big)
   ]
3. 理解为什么叫 SARSA：一条更新用到了五元组
   ((S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}))。
4. 理解“on-policy”的含义：目标里用的 (a_{t+1}) 是按当前行为策略（如 ε-greedy）采样出来的。

---

### 1) 从 TD(0) 过渡：把 (V) 换成 (Q)

TD(0) 评估 (V^\pi) 的目标是：
[
r_{t+1}+\gamma V(s_{t+1})
]
但做控制时，最重要的是“动作选择”，所以我们把价值函数换成动作价值：

* 状态价值：只问“在 s 有多好”
* 动作价值：问“在 s 选 a 有多好”，这就能直接用于改策略（贪心/ε-greedy）。

于是 SARSA 的目标是“一步 bootstrap 的动作价值”：
[
\text{target}=r_{t+1}+\gamma Q(s_{t+1},a_{t+1})
]

---

### 2) 为什么 SARSA 用 (Q(s',a')) 而不是 (\max_{a'}Q(s',a'))

因为 SARSA 是 **on-policy**：它评估的是“当前实际会执行的策略”的 (Q^\pi)。
如果你的行为策略是 ε-greedy，那么下一步你会以一定概率探索（选非最优动作）。SARSA 的目标就必须反映这种“真实会发生的下一动作”，因此用 (a_{t+1}\sim\pi(\cdot|s_{t+1}))。

对比（先记结论，后面深入）：

* **SARSA**：目标用 (Q(s',a'))，其中 (a') 按当前策略采样（更保守、更贴近行为）
* **Q-learning**：目标用 (\max_{a'}Q(s',a'))（off-policy，直接逼近最优）

---

### 3) SARSA 如何“优化策略”

SARSA 不是直接写出最优方程求解，而是交替发生两件事：

1. **用当前策略产生数据并更新 (Q)**（近似评估当前策略）
2. **用更新后的 (Q) 改变策略（ε-greedy）**（近似策略改进）

这就是“广义策略迭代”（Generalized Policy Iteration, GPI）的雏形：评估与改进交织进行。

---


# 第 9 课（9.2）：Q-learning——为什么用 (\max)（off-policy TD control）

上一节 SARSA 的更新目标里用的是下一步“实际会执行的动作” (a_{t+1})。本节只讲一个点：**Q-learning 把这个动作替换成 (\arg\max) 的贪心动作，从而直接逼近最优动作价值 (Q^*)**。这一步改变，带来的就是 “off-policy”。


1. 写出 Q-learning 的更新式并能逐项解释：
   [
   Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\Big(r_{t+1}+\gamma \max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\Big)
   ]
2. 明确它在逼近的对象：**贝尔曼最优方程对应的固定点 (Q^*)**。
3. 理解 “off-policy” 的精确定义：

   * **行为策略**（behavior policy）用于采样数据（通常 ε-greedy）
   * **目标策略**（target policy）在更新目标中出现（这里是纯贪心 (\arg\max)）
     两者不必相同，因此是 off-policy。
4. 能用一句话对比 SARSA vs Q-learning：

   * SARSA：目标用 (Q(s',a'))（(a') 来自行为策略）
   * Q-learning：目标用 (\max_{a'}Q(s',a'))（来自目标策略）

---

### 1) Q-learning 的“唯一关键改动”

SARSA：
[
\text{target}=r+\gamma Q(s',a') \quad (a'\sim\pi)
]
Q-learning：
[
\text{target}=r+\gamma \max_{a'}Q(s',a')
]
这相当于把“下一步会做什么”替换成“下一步如果我总做最好的，会怎样”。因此它在学习的是最优值函数而不是某个具体 ε-greedy 策略的值函数。

---

### 2) 为什么这对应贝尔曼最优方程

最优动作价值满足：
[
Q^*(s,a)=\mathbb{E}\big[r_{t+1}+\gamma \max_{a'}Q^*(s_{t+1},a')\mid s,a\big]
]
Q-learning 用样本把右侧期望替换成单次采样，再做随机逼近更新，所以它是“在数据上逼近最优固定点”。

---

### 3) off-policy 的直觉（非常重要）

你采样数据时可能会探索（ε-greedy），但更新目标假设你将来会按纯贪心行动。
因此：

* 数据来自 “探索中的你”（行为策略）
* 目标来自 “理想贪心的你”（目标策略）

这就是 off-policy。

---

# 第 9 课（9.3）：Cliff Walking 中 SARSA 更“保守”、Q-learning 更“激进”——原因与可复现实验

本节只讲一个点：**在存在“悬崖/高风险区域 + 持续探索（ε-greedy）”时，SARSA（on-policy）会学出对探索更鲁棒的安全路线，而 Q-learning（off-policy）倾向学出贴悬崖的最短路线；在训练过程中（行为仍带 ε 探索）两者表现会出现系统性差异。**

---

1. 用 CliffWalking 场景解释：**“目标策略”与“行为策略”不一致会如何影响学习到的路线**。
2. 明确差异来自更新目标：

   * SARSA：(r+\gamma Q(s',a'))，其中 (a'\sim) 行为策略（含探索）
   * Q-learning：(r+\gamma\max_{a'}Q(s',a'))，相当于目标策略为纯贪心
3. 理解“训练回报曲线”常见结论：**在固定 ε 探索下，SARSA 通常比 Q-learning 的平均回报更好（更少掉崖），但最终学到的贪心路径可能更长更安全**。

---

### 1) CliffWalking 环境的关键结构（为什么它是“放大镜”）

* 每走一步代价：(-1)
* 走到悬崖（cliff）：强惩罚 (-100)，并被送回起点（episode 不中止）
* 目标在悬崖右侧：最短路通常贴着悬崖边走（风险极高）

这使得“探索”本身就有显著代价：只要在悬崖边上 ε 随机抖一下，很容易掉崖。

---

### 2) SARSA 为什么更保守（核心：它在学“带探索的真实策略”）

SARSA 更新用 (a')（下一步实际采样到的动作）：
[
Q(s,a)\leftarrow Q(s,a)+\alpha\Big(r+\gamma Q(s',a')-Q(s,a)\Big)
]
如果你的行为策略是 ε-greedy，那么在悬崖边：

* 你有 ε 概率会执行“错误动作”导致掉崖
* SARSA 的目标里会把这种风险纳入期望（因为 (a') 的分布就是含探索的分布）

因此 SARSA 倾向于学出**离悬崖更远**的路线，使得“即便探索抖一下也不至于立刻掉崖”。

---

### 3) Q-learning 为什么更激进（核心：它在学“假设未来总最优”）

Q-learning 更新用 max：
[
Q(s,a)\leftarrow Q(s,a)+\alpha\Big(r+\gamma \max_{a'}Q(s',a')-Q(s,a)\Big)
]
它的目标隐含假设：到了 (s') 之后你会执行最优动作（纯贪心），不考虑 ε 探索的失误风险。
因此它更倾向于学出“**最短**且贴悬崖”的路线。

注意：这不意味着 Q-learning “错了”。它逼近的是最优值函数 (Q^*)。问题在于：**如果你训练/执行时始终保留 ε 探索，那么行为表现并不等同于纯贪心策略的表现**。

---


# 第 9 课（9.4）：Q-learning 的过估计偏差（Maximization Bias）与 Double Q-learning

本节只讲一个点：**为什么 Q-learning 里的 (\max_{a'}Q(s',a')) 会系统性“偏大”（过估计），以及 Double Q-learning 用什么最小改动缓解它。**

---

1. 解释“过估计偏差”的来源：**对含噪估计取最大，会把噪声的正偏部分挑出来**。
2. 写出 Q-learning 的目标与偏差点：
   [
   \text{target}=r+\gamma\max_{a'}Q(s',a')
   ]
3. 写出 Double Q-learning 的核心更新（选择与评估分离）：
   [
   a^*=\arg\max_{a'}Q_1(s',a'),\quad
   \text{target}=r+\gamma Q_2(s',a^*)
   ]
   （或对称地交换 (Q_1,Q_2)）

---

### 1) 过估计偏差的“最核心一句话”

即使每个动作价值估计 ( \hat Q(s',a)) 都是“无偏”的（围绕真实值上下抖动），**(\max_a \hat Q(s',a)) 也往往是“有正偏”的**。

直觉原因：

* (\max) 会倾向性选中“恰好被噪声抬高”的那个动作
* 所以最大值不仅包含真实值，还包含被挑出来的正噪声

一个极简直觉例子：
若两个动作真实值都为 0，估计为 (X,Y)，且 (\mathbb{E}[X]=\mathbb{E}[Y]=0)。通常会有：
[
\mathbb{E}[\max(X,Y)] > 0
]
这就说明“取最大”本身会引入正偏。

---

## 微课 9.4（理论）：Double Q-learning 如何缓解？

### 2) Double Q 的核心思想：**把“选动作”和“评估动作价值”分开**

Q-learning 的偏差来自同一套 noisy 估计同时做了两件事：

1. 选最大动作（argmax）
2. 用最大动作的估计值（max）做评估

Double Q-learning 维护两套估计 (Q_1,Q_2)，每次随机更新其中之一：

* 以 0.5 概率更新 (Q_1)：
  [
  a^*=\arg\max_{a'}Q_1(s',a'),\quad
  \text{target}=r+\gamma Q_2(s',a^*)
  ]
* 否则更新 (Q_2)（对称交换）

这样做的效果：

* argmax 的“挑噪声”主要发生在 (Q_1) 上
* 评估值来自相对独立的 (Q_2)，不会同步被同一个噪声抬高
  从而显著减弱过估计。

---

# 微课 9.4（编码）：用 Sutton 经典“Maximization Bias”小 MDP 复现实验

这个实验非常适合“把偏差现象看清楚”，且代码短、逻辑清晰。

**环境结构：**

* 状态 A：两动作

  * LEFT：直接终止，奖励 0
  * RIGHT：转移到状态 B，奖励 0
* 状态 B：有很多动作（比如 10 个），每个动作都会终止，但奖励是噪声（例如 (N(0,1))），期望为 0
  因此：**从 A 无论选 LEFT 还是 RIGHT，真实期望回报都是 0**。

但 Q-learning 在 B 上要做 (\max)，会把噪声奖励“挑大”，导致它错误地认为走 RIGHT 更好，进而在 A 上更倾向选 RIGHT。Double Q-learning 会明显缓解这个倾向。

---
# 第 9 课（9.5）：Expected SARSA——用“对策略的期望”替代采样或 max，让更新更稳定

本节只讲一个点：**把 SARSA 的目标从“采样到的下一动作”改成“对下一动作分布的期望”**。这会显著降低目标的方差；同时它仍然是 on-policy（用的就是当前行为策略的分布）。

---

## 微课 9.5（理论）：Expected SARSA 的目标到底是什么

### 教学目标

学完你应能：

1. 写出 Expected SARSA 的更新式，并能解释它与 SARSA/Q-learning 的差别。
2. 理解它为什么更稳定：**去掉了对 (a') 的采样噪声（用期望替代）**。
3. 能在 ε-greedy 策略下，明确写出 (\sum_{a'}\pi(a'|s')Q(s',a')) 的具体计算方式。

---

### 1) 对比三种目标（把差异压到一个公式里）

**SARSA（采样下一动作）**
[
Q(s,a)\leftarrow Q(s,a)+\alpha\Big(r+\gamma Q(s',a')-Q(s,a)\Big),\ \ a'\sim\pi(\cdot|s')
]

**Q-learning（假设未来最优，max）**
[
Q(s,a)\leftarrow Q(s,a)+\alpha\Big(r+\gamma \max_{a'}Q(s',a')-Q(s,a)\Big)
]

**Expected SARSA（对策略取期望）**
[
\boxed{
Q(s,a)\leftarrow Q(s,a)+\alpha\Big(r+\gamma \sum_{a'}\pi(a'|s')Q(s',a')-Q(s,a)\Big)
}
]

一句话定位：

* SARSA：目标随机（因 (a') 采样）
* Expected SARSA：目标更“确定”（对 (a') 做期望）
* Q-learning：目标更“激进”（用 max）

---

### 2) 为什么 Expected SARSA 更稳定（只讲关键原因）

在 SARSA 中，目标项 (Q(s',a')) 取决于随机采样的 (a')，所以 TD 误差的方差更大。
Expected SARSA 把这一项替换为加权平均：
[
\mathbb{E}_{a'\sim\pi}[Q(s',a')]
]
因此减少了采样噪声，通常能得到更平滑的学习曲线。

---

### 3) ε-greedy 下的期望怎么计算（你必须会算）

设动作集合大小为 (|\mathcal{A}|)。ε-greedy 策略在 (s') 处：

* 以 (1-\epsilon) 选贪心动作 (a_g=\arg\max_a Q(s',a))
* 以 (\epsilon) 做均匀随机：每个动作概率 (\epsilon/|\mathcal{A}|)

则：
[
\pi(a|s')=
\begin{cases}
1-\epsilon+\epsilon/|\mathcal{A}|, & a=a_g \
\epsilon/|\mathcal{A}|, & a\neq a_g
\end{cases}
]

于是期望项为：
[
\sum_{a'}\pi(a'|s')Q(s',a')
]
在离散小动作空间里，这是非常便宜的：遍历动作做一次加权和，复杂度 (O(|\mathcal{A}|))。

---
