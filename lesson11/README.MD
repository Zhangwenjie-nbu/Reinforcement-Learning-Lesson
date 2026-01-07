# 第 11 课 11.1：DQN 为什么必须要有 Replay Buffer 与 Target Network（本节只讲这一个稳定性问题）

你已经学到“致命三元组”：**函数逼近 + 自举 + off-policy** 会带来结构性不稳定。DQN 正是这三者的组合（用神经网络逼近 (Q)，用 TD target 自举，且用 ε-greedy 行为数据做离策略学习），所以 DQN 的两项核心工程机制——**经验回放（Replay Buffer）与目标网络（Target Network）**——本质上是在“破坏不稳定的正反馈回路”。

---
1. 说清楚 DQN 的更新为什么天然“不稳定”：目标在动、数据分布在动、并且是 off-policy bootstrapping。
2. 精确解释 Replay Buffer 在做什么、为什么能稳定：**去相关 + 近似 i.i.d. + 复用样本**。
3. 精确解释 Target Network 在做什么、为什么能稳定：**把自举目标暂时冻结，降低目标漂移**。
4. 能把这两者与“致命三元组”的三条链路一一对应起来（破坏哪条反馈）。

---

### 1) 先写出 DQN 的核心更新（你必须能对上号）

DQN 学的是动作价值 (Q_\theta(s,a))。典型的平方 TD 误差损失：

[
L(\theta)=\mathbb{E}\Big[\big(y - Q_\theta(s,a)\big)^2\Big]
]

其中目标 (y) 是自举的：

[
y = r + \gamma \max_{a'} Q_{\theta^-}(s',a')
]

关键点：

* (Q_\theta) 用神经网络逼近（函数逼近）
* (y) 里用到了 (Q)（自举）
* 数据 ((s,a,r,s')) 来自行为策略（off-policy）

如果你把 (\theta^-=\theta)（不用 target net），那就变成：
[
y = r + \gamma \max_{a'} Q_{\theta}(s',a')
]
此时 **同一个网络既在“当学生”，又在“当老师给标签”**，而且老师的标签会随着学生参数更新而立刻改变——这就是“移动目标（moving target）”的核心不稳定来源之一。

---

### 2) Replay Buffer：它到底解决了什么

#### 现象：在线数据强相关

强化学习按时间顺序收集到的样本是：
[
(s_t,a_t,r_{t+1},s_{t+1}),\ (s_{t+1},a_{t+1},r_{t+2},s_{t+2}),\dots
]
这些样本高度相关（相邻状态几乎连续变化），用它们做 SGD 会导致：

* 梯度方向抖动更强
* 更容易把网络推到“对近期轨迹过拟合”的方向
* 训练不稳定

#### Replay Buffer 做的事（两句话）

1. **把连续交互数据存起来**
2. **训练时从缓冲区随机采样一批（mini-batch）**

这带来三个直接效果：

* **去相关**：随机采样近似 i.i.d.，更符合 SGD 假设
* **分布更平稳**：不只看最新几步
* **样本复用**：同一条经验可训练多次，提升数据效率

> 对致命三元组的对应关系：Replay Buffer 主要在缓解 “off-policy + function approximation” 下由于**数据相关性与分布漂移**引发的不稳定，但它本身并不能消除自举的 moving target。

---

### 3) Target Network：它到底解决了什么

Target Network 维护一份“慢更新”的参数 (\theta^-)，用于计算 TD target：

[
y = r + \gamma \max_{a'} Q_{\theta^-}(s',a')
]

并且每隔 (K) 步才同步一次：
[
\theta^- \leftarrow \theta
]

直觉：让“老师”暂时固定一段时间，学生在这段时间里去拟合一个相对稳定的目标，避免每一步更新都把目标一起挪走。

> 对致命三元组的对应关系：Target Network 主要在缓解 “bootstrapping + function approximation” 下的**正反馈**（moving target），让自举目标变化更慢、更可控。

---

# 第 11 课 11.2：DQN 的过估计偏差（Overestimation Bias）与 Double DQN 的修复机制

本节只讲一个点：**为什么标准 DQN 的 (\max) 目标会系统性偏大，以及 Double DQN 如何通过“选动作”和“评估动作”解耦来减小偏差**。最后用一个极小环境复现：标准 DQN 更容易“被虚高的 Q 值骗走”，Double DQN 明显缓解。

---
1. 精准解释过估计偏差的来源：(\max) 作用在带噪估计上导致正偏。
2. 写出标准 DQN 与 Double DQN 的目标（target）公式，并说明差异。
3. 通过可运行实验观察：标准 DQN 更倾向选择“实际上不更好”的动作；Double DQN 更接近无偏。

---

## 1) 过估计偏差从哪里来（核心一句话讲透）

假设某个状态下多个动作的真实价值都一样（例如都为 0），但你的网络估计值带噪声：
[
\hat Q(a) = Q(a) + \epsilon(a)
]
那么：
[
\mathbb{E}\left[\max_a \hat Q(a)\right] ;=; \mathbb{E}\left[\max_a (Q(a)+\epsilon(a))\right] ;\ge; \max_a Q(a)
]
**不等号的原因**：(\max) 会更偏向挑到“噪声为正的那一个”，所以期望会偏大。这是一个结构性偏差，不是训练没收敛、也不是超参没调好。

---

## 2) 标准 DQN vs Double DQN：目标函数的差异

### 标准 DQN 的 TD 目标

[
y = r + \gamma \max_{a'} Q_{\theta^-}(s',a')
]
这里 **选动作（argmax）** 和 **评估动作（Q 值）** 都来自同一个网络（目标网络 (\theta^-)）。

### Double DQN 的 TD 目标

[
a^* = \arg\max_{a'} Q_{\theta}(s',a')
]
[
y = r + \gamma Q_{\theta^-}(s',a^*)
]
解释：

* 用 **online 网络 (\theta)** 负责“选动作”（更贴近当前策略）
* 用 **target 网络 (\theta^-)** 负责“评估该动作的 Q 值”（更稳定、也减小 (\max) 噪声偏差）

直观上：Double DQN 让“挑最大值”和“给这个最大值打分”来自不同估计器，降低“挑到噪声最大者”的系统偏差。

---
# 第 11 课 11.3：Dueling DQN——把 (Q(s,a)) 拆成 (V(s)) 与 (A(s,a))，为什么能更快、更稳

本节只讲一个点：**Dueling 架构的“表示方式”如何改变学习难度**。它不改算法（还是 DQN 的 TD 学习），只改网络结构，但在“很多状态下动作差别不大、或动作很多且大量动作无意义”的问题上，通常能显著提升样本效率与稳定性。

---

1. 解释：为什么直接学 (Q(s,a)) 会“浪费容量”与“学习慢”。
2. 写出 Dueling 的分解：
   [
   Q(s,a)=V(s)+A(s,a)
   ]
   并理解“不可辨识性（identifiability）”问题为何必须处理。
3. 说明常用聚合（aggregation）形式：
   [
   Q(s,a)=V(s)+\Big(A(s,a)-\frac{1}{|\mathcal A|}\sum_{a'}A(s,a')\Big)
   ]
4. 在代码里实现 DuelingQNetwork，并在一个“动作很多但多数动作信息量低”的极小环境中，对比标准 DQN 与 Dueling DQN 的学习曲线。

---

## 1) 为什么要 Dueling：很多时候你更容易先学到“这个状态好不好”

标准 DQN 的网络直接输出所有动作的 (Q(s,a))。但在很多任务里：

* **大量状态**：选哪个动作差别不大（动作优势很小），你真正需要的是“这个状态整体值高不高”（即 (V(s))）。
* **大量动作**：很多动作几乎等价、或大多数动作是无意义的（例如大量 no-op），但标准网络仍需要为每个动作学一套 Q 输出，学习信号被“稀释”。

Dueling 的关键直觉是：

> 把“状态本身的价值”与“动作带来的相对增益”分开学，让网络更快地把容量用在最该学的东西上。

---

## 2) 分解：(Q(s,a)=V(s)+A(s,a)) 但会遇到不可辨识性

如果你只写：
[
Q(s,a)=V(s)+A(s,a)
]
那么存在无穷多个等价解：对任意常数 (c)，
[
(V(s)+c) + (A(s,a)-c)
]
给出同样的 (Q)。这意味着训练过程会出现“漂移自由度”，导致数值不稳定或学习效率差。

### 常用解决：强制优势函数零均值

最常用聚合方式：
[
Q(s,a)=V(s)+\Big(A(s,a)-\text{mean}_{a'}A(s,a')\Big)
]
这样就固定了表示：优势在动作维度上均值为 0，(V(s)) 就成为“动作平均意义下的状态价值”。

---

## 3) Dueling 到底什么时候更有效（你需要能判断适用性）

更可能受益的场景：

* 动作多（|A|大），且很多动作差别小/冗余；
* 大量状态下动作选择不敏感（“先搞清楚状态值”更重要）；
* 或存在很多无意义动作（no-op、弱动作、重复动作）。

不一定受益甚至可能差不多的场景：

* 动作极少（例如只有 2 个动作且差异很大）；
* 每个状态动作差别都非常关键且优势结构简单（标准 DQN 也不太难学）。

---
# 第 11 课 11.4：Prioritized Experience Replay（PER）——为什么更快，以及为什么必须做重要性采样校正（IS correction）

本节只讲一个点：**PER 用 TD-error 来“更常抽到更有学习价值的样本”，但这样会改变采样分布，带来偏差；因此必须用重要性采样权重（IS weights）把更新改回近似无偏。**

---
1. 解释：为什么“均匀采样 replay”在很多任务里浪费算力。
2. 写出 PER 的核心采样概率与优先级定义。
3. 精确说明：PER 引入了什么偏差，IS correction 如何修正（以及 (\beta) 为什么要退火到 1）。
4. 在代码里实现一个可运行的 PERBuffer，并在 DQN 训练中：

   * 用 IS weights 做 **加权 TD loss**
   * 用 batch 的 TD-error **回写更新 priorities**

---

## 1) 为什么均匀采样会“浪费”

Replay buffer 里很多 transition 可能已经“学会了”（TD-error 很小），你再抽它们训练，梯度几乎为 0，等于浪费 batch。

PER 的直觉：

> **抽“目前学得不好的样本”**（TD-error 大的）能更快减少总体误差。

---

## 2) PER 的核心公式（你必须记住的三行）

对第 (i) 条经验，定义优先级（常用）：
[
p_i = |\delta_i| + \varepsilon
]
其中 (\delta_i) 是 TD-error，(\varepsilon>0) 避免 0 概率。

采样概率（带指数 (\alpha) 控制“偏向程度”）：
[
P(i)=\frac{p_i^\alpha}{\sum_j p_j^\alpha},\quad \alpha\in[0,1]
]

* (\alpha=0) 退化为均匀采样
* (\alpha) 越大越偏向“大误差样本”

---

## 3) PER 为什么会“引入偏差”，IS correction 做了什么

DQN 的期望损失本质上假设你从某个分布（近似 buffer 的均匀分布）采样做 SGD。

PER 改成 (P(i)) 后，相当于你在最小化一个**被重加权的目标**（偏差来源），会导致估计偏移，甚至在后期收敛时产生不稳定（因为你持续过度关注少数大误差样本）。

### IS correction（重要性采样校正）

用权重把采样分布校正回“近似均匀”：
[
w_i = \left(\frac{1}{N}\cdot\frac{1}{P(i)}\right)^\beta
]

* (N) 是 buffer 当前大小
* (\beta\in[0,1]) 控制校正强度
* (\beta=1) 时是完整校正（理论上更接近无偏）
* 实践中通常 **(\beta) 从小到大退火到 1**：早期允许有偏以加速学习，后期逐步变“更无偏”以稳定收敛。

训练时用加权 MSE：
[
L=\mathbb{E}*{i\sim P}\left[w_i,(y_i-Q*\theta(s_i,a_i))^2\right]
]
通常还会把 (w_i) 除以 (\max_i w_i) 做归一化，避免数值爆炸。

---
# 第 11 课 11.5：Noisy Networks（NoisyNet）——用“参数噪声”替代 ε-greedy，形成更一致的探索

本节只讲一个点：**为什么“在网络参数里注入可学习噪声”可以替代 ε-greedy，并带来更一致、状态相关、时间相关的探索**；以及如何在你现有 DQN 脚手架里做**最小改动**实现 NoisyNet（NoisyLinear）。

---
1. 清晰解释：NoisyNet 与 ε-greedy 的探索机制差异（“动作噪声” vs “参数噪声”）。
2. 理解 NoisyLinear 的形式：(\mu)（均值参数） + (\sigma)（噪声强度参数）× (\varepsilon)（随机噪声）。
3. 理解为什么 **(\sigma) 是可学习的**：网络会自动决定“哪里需要探索、探索多强”。
4. 在代码里实现 `NoisyLinear`（Factorized Gaussian Noise 版本），并把 QNetwork 替换为 NoisyQNetwork，让 DQN 在 **ε=0** 情况下仍能探索并学习。

---

## 1) ε-greedy 的本质与局限

ε-greedy 是“动作空间层面”的随机：

* 以概率 ε 随机选动作；以概率 (1-\varepsilon) 选 (\arg\max_a Q(s,a))。

它的问题（不是“错”，而是结构限制）：

* **探索是状态无关的**：在关键状态与无关状态都同样随机。
* **探索是时间不一致的**：每一步独立抖动，难以形成“持续尝试某条策略”的一致性（尤其在长时序/稀疏奖励任务里）。
* ε 的手工退火需要调参：退火太快会早熟，太慢会浪费样本。

---

## 2) NoisyNet 的核心思想：让策略随机性来自“参数扰动”

NoisyLinear 把线性层参数写成：
[
W = W_\mu + W_\sigma \odot \varepsilon_W,\quad b = b_\mu + b_\sigma \odot \varepsilon_b
]
其中：

* (W_\mu, b_\mu)：可学习的“均值参数”（相当于普通网络参数）
* (W_\sigma, b_\sigma)：可学习的“噪声幅度参数”（决定探索强度）
* (\varepsilon)：随机噪声（每次重采样改变“本次策略”）

**为什么这能更一致地探索？**
因为一次噪声采样会同时影响很多动作的 Q 输出，导致策略在一段时间里更像“换了一套略不同的 Q 函数”，这更接近“尝试一种不同策略”（而不是每步随手乱点一下）。

---

## 3) Factorized Gaussian Noise（教学上推荐实现）

直接给每个权重采样一个噪声矩阵成本高。Factorized 方式用两个向量生成矩阵噪声：

* 采样 ( \epsilon_{in}\in\mathbb{R}^{in}, \epsilon_{out}\in\mathbb{R}^{out} )
* 用 ( f(x)=\mathrm{sign}(x)\sqrt{|x|} ) 变换后做外积：
  [
  \varepsilon_W = f(\epsilon_{out}) \cdot f(\epsilon_{in})^\top,\quad \varepsilon_b=f(\epsilon_{out})
  ]
  这会显著降低采样与存储成本，是常用实现。

---

# 微课 11.5（编码）：最小 NoisyNet DQN（对比 ε-greedy vs NoisyNet）

你将运行一个脚本，做两次训练：

* **Run A：标准 DQN（ε-greedy）**
* **Run B：NoisyNet DQN（ε=0，仅靠参数噪声探索）**

环境用你已经熟悉的链式任务（ChainMDP），并保留 **Replay + Target**（稳定性机制不变）。

---
# 第 11 课 11.6：n-step return（多步回报）——为什么能显著加速 DQN 的学习传播

本节只讲一个点：**把 1-step TD target 换成 n-step target，能更快把稀疏/延迟奖励“往回传播”，从而加速学习**。我们会把它落到代码上：只改 replay 存储方式与 TD target 的公式，其余（Replay + Target）保持不变。

---
1. 写出 1-step 与 n-step 的 target 公式，并解释区别。
2. 理解 n-step 的核心收益：**更快传播奖励**（credit assignment 更直接）。
3. 理解代价：n 越大越接近 MC，**方差更大**；且在 off-policy 设定下理论更复杂（实践中仍常用）。
4. 在 DQN 中实现 n-step：

   * replay 存储从“单步 transition”改为“n-step transition”
   * target 改为：(R^{(n)} + \gamma^n \max_a Q(s_{t+n},a))

---

### 1) 1-step TD（你已经很熟）

标准 DQN 的 1-step 目标：
[
y^{(1)} = r_{t+1} + \gamma \max_a Q_{\theta^-}(s_{t+1},a)
]
问题：如果奖励要很多步后才出现（延迟奖励），那么“正奖励信息”要一层层往回传，传播速度慢。

---

### 2) n-step return：一次性看 n 步

定义 n-step 累积回报：
[
R^{(n)}*t = \sum*{k=0}^{n-1} \gamma^k r_{t+1+k}
]
对应的 n-step target：
[
y^{(n)} = R^{(n)}*t + \gamma^n \max_a Q*{\theta^-}(s_{t+n},a)
]
如果在第 (t+m)（(m<n)）步就终止，则后面不再 bootstrap，目标退化为到终止为止的折扣和。

**直觉**：你把“未来 n 步的真实奖励”直接纳入监督信号，相当于把监督更靠近真实回报，奖励传播更快。

---

### 3) 偏差/方差权衡（只记结构）

* (n=1)：偏差较大（自举多），方差较小（用估计代替真实未来）。
* (n) 增大：偏差下降、方差上升；极端 (n=) episode 长度接近 Monte Carlo。
* 在 DQN 实战中常用 (n=3) 或 (n=5)（Rainbow 也用 multi-step），通常是“传播速度”和“噪声”之间的折中。

---