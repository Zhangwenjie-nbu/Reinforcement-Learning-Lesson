# 第 10 课（10.1）：为什么需要函数逼近——从“表格 V(s)”到“参数化 (V_\theta(s))”的最小跨越

本节只讲一个点：**表格方法为什么在大状态空间不可行，以及最简单的价值函数逼近形式 (V_\theta(s)=\phi(s)^\top\theta) 是什么、怎么和 TD(0) 结合。**

---

1. 明确表格法的硬限制：状态数大/连续时无法存储与充分采样。
2. 写出最基本的函数逼近形式：
   [
   V_\theta(s)=\phi(s)^\top\theta
   ]
   并解释 (\phi(s))（特征）与 (\theta)（参数）的角色。
3. 写出线性 TD(0) 的更新（半梯度）：
   [
   \delta_t=r_{t+1}+\gamma V_\theta(s_{t+1})-V_\theta(s_t)
   ]
   [
   \theta \leftarrow \theta + \alpha,\delta_t,\phi(s_t)
   ]
4. 理解为什么叫“半梯度（semi-gradient）”：目标里含 (V_\theta(s_{t+1})) 但我们不对它反传梯度（这是 TD 的典型做法）。

---

### 1) 表格方法为什么不够用（只讲最核心两条）

* **存储不可行**：若状态空间是图像、连续位置、组合结构，(|\mathcal{S}|) 极大甚至无限，无法维护 (V[s]) 或 (Q[s,a]) 表。
* **泛化能力缺失**：表格法对“没见过的状态”没有任何推断能力；而函数逼近可以在相似状态之间共享参数，实现泛化。

---

### 2) 最小的函数逼近：线性模型

把每个状态映射为一个特征向量 (\phi(s)\in\mathbb{R}^d)，例如：

* one-hot（退化为表格）
* 位置/距离等手工特征
* tile coding / RBF（后面会讲）
* 神经网络输出特征（更后面）

然后用一个参数向量 (\theta\in\mathbb{R}^d) 表示价值：
[
V_\theta(s)=\phi(s)^\top\theta
]
直觉：(\theta) 存的是“特征权重”，你学的是这些权重，而不是每个状态一个值。

---

### 3) 线性 TD(0) 的更新从哪里来（一步推导直觉）

TD(0) 想让：
[
V_\theta(s_t) \approx r_{t+1}+\gamma V_\theta(s_{t+1})
]
定义 TD 误差：
[
\delta_t=r_{t+1}+\gamma V_\theta(s_{t+1})-V_\theta(s_t)
]
如果把 (\frac{1}{2}\delta_t^2) 看作损失，对 (V_\theta(s_t)) 做梯度下降（只对当前预测项求导），有：
[
\nabla_\theta V_\theta(s_t)=\phi(s_t)
]
于是得到半梯度更新：
[
\theta \leftarrow \theta + \alpha,\delta_t,\phi(s_t)
]

---


# 第 10 课（10.2）：Tile Coding（平铺编码）——让线性方法在连续状态上“像表格一样好用”，但又能泛化

本节只讲一个点：**Tile Coding 是什么、为什么它在强化学习里很常用，以及如何把它接到线性 TD(0) 上完成连续状态的价值评估。**

---

1. 解释 Tile Coding（也叫 coarse coding / CMAC）的基本构造：**多组错位网格（tilings）+ 稀疏二值特征**。
2. 明确它解决什么问题：连续状态无法表格化，但我们仍想要“局部泛化、训练稳定、计算便宜”。
3. 写出用 Tile Coding 表示的线性价值函数：
   [
   V_\theta(s)=\sum_{i\in \text{active}(s)} \theta_i
   ]
4. 知道 TD(0) 如何更新这些权重（并理解为什么常把步长除以 tilings 数）。

---

### 1) 线性函数逼近的难点：手工特征往往“全局耦合”

上一课我们用 ([x, x^2, 1]) 这类特征做逼近，它的问题是：

* 一个参数会影响所有状态（全局耦合）
* 学习信号在全空间传播，容易欠拟合/震荡
  你通常更希望：**只影响“附近的状态”**，让学习变得像表格那样“局部更新”。

---

### 2) Tile Coding 的做法：把连续空间“切格子”，并且切很多次（错位）

以 1D 连续状态 (x\in[0,1]) 为例：

* 先把区间切成 (N) 个 tile（格子）：这是一套 tiling
* 再来第二套 tiling，但整体平移一点点（offset）
* 再来第三套……共 (M) 套 tilings

于是对于任意一个 (x)，每套 tiling 都会命中一个 tile。
所以 **active 特征数 = tilings 数 (M)**，并且这些特征是 **稀疏二值**（命中为 1，其余为 0）。

这就产生了一个关键效果：

* 单个 tiling 很粗（泛化强但不精）
* 多个错位 tilings 叠加后，整体分辨率变细（更精）

你可以把它理解成：**“用很多把略微错位的粗尺子一起量长度”**。

---

### 3) Tile Coding 的价值函数形式为什么特别适合 TD

因为它是稀疏的：
[
V_\theta(x)=\sum_{i\in \text{active}(x)} \theta_i
]
每次更新只动少量参数（正好是 (M) 个），和表格更新一样“局部、便宜、稳定”。

---

### 4) 为什么实践里常用 (\alpha / M)

因为每个状态会激活 (M) 个特征。如果你直接用 (\alpha) 更新所有激活权重，相当于“有效步长”被放大了约 (M) 倍。
一个常见做法是：

[
\theta_i \leftarrow \theta_i + \frac{\alpha}{M},\delta
\quad \text{for } i\in\text{active}(x)
]

这样当你改变 tilings 数时，整体更新幅度更可控。

---

# 第 10 课（10.3）：从 (V_\theta(s)) 到 (Q_\theta(s,a))——Tile Coding + Semi-gradient SARSA（连续状态下的控制）

上一节你已经能用 Tile Coding + 线性 TD(0) 评估 (V^\pi(x))。本节只讲一个点：**把“状态价值”扩展为“动作价值”，并用 Semi-gradient SARSA 在连续状态环境里学会“朝右走”**。

---

1. 明确控制问题需要 (Q(s,a))：因为策略改进依赖“比较动作”。
2. 写出 Semi-gradient SARSA 的核心更新：
   [
   \delta_t=r_{t+1}+\gamma Q_\theta(s_{t+1},a_{t+1})-Q_\theta(s_t,a_t)
   ]
   [
   \theta \leftarrow \theta+\alpha,\delta_t,\nabla_\theta Q_\theta(s_t,a_t)
   ]
3. 在 Tile Coding（稀疏二值特征）下，把 (\nabla_\theta Q_\theta) 具体化为“只更新激活权重”。
4. 理解一个关键工程点：**如何构造 (\phi(s,a))**（最简单、最常用的做法：每个动作一套独立的特征权重）。

---

### 1) 从 (V) 到 (Q)：控制为什么绕不开动作价值

* 评估：给定 (\pi)，学 (V^\pi(s))，只回答“在 s 好不好”。
* 控制：要改进策略，需要知道“在 s 选哪个动作更好”。因此必须学：
  [
  Q^\pi(s,a)=\mathbb{E}[G_t\mid s_t=s,a_t=a]
  ]
  有了 (Q)，最直接的策略改进就是贪心或 ε-greedy：
  [
  \pi(a|s)\approx \arg\max_a Q(s,a)
  ]

---

### 2) Semi-gradient SARSA（函数逼近版）的更新是什么

表格 SARSA：
[
Q(s,a)\leftarrow Q(s,a)+\alpha\delta_t
]
函数逼近（参数化）：
[
Q_\theta(s,a) \approx Q(s,a)
]
Semi-gradient SARSA 的更新写成：
[
\theta \leftarrow \theta+\alpha,\delta_t,\nabla_\theta Q_\theta(s_t,a_t)
]
其中
[
\delta_t=r_{t+1}+\gamma Q_\theta(s_{t+1},a_{t+1})-Q_\theta(s_t,a_t)
]

“semi-gradient”的含义：TD 目标里也有 (Q_\theta)，但我们不把梯度穿过目标项（这是 TD 的典型做法）。

---

### 3) Tile Coding 下 (\phi(s,a)) 怎么构造（最简单最实用）

我们上一节的 tile coding 给的是 (\phi(s))：一个状态激活 (M) 个 tile。

要变成 (\phi(s,a))，最常用做法是：**每个动作拥有一套独立的 tile 权重**。

如果状态 tile 特征总数是 (F)，动作数是 (|A|)，那么总参数数为：
[
|A|\cdot F
]
具体编码：

* 状态激活特征 ids：`feats = active_features(x)`（长度 M）
* 动作 a 的 state-action 特征 ids：`a*F + feat`（把索引整体平移）

这样：
[
Q_\theta(s,a)=\sum_{i\in active(s,a)} w_i
]
梯度就是对这些激活权重为 1，其余为 0，因此更新等价于：

* 只更新激活的那 M 个权重

同样，常用 (\alpha/M) 缩放，避免 tilings 变多导致更新过猛。

---

## 第10课 （10.3）：致命三元组（Deadly Triad）与“为什么会发散”——Baird 反例（可复现实验）

1. **精确定义**“致命三元组”三件事分别是什么：函数逼近 / 自举 / 离策略（off-policy）。
2. **理解关键结论**：三者同时出现时，常见的半梯度（semi-gradient）TD 类更新**可能发散**，这不是“随机性导致的不稳定”，而是算法与目标/分布不匹配带来的结构性风险。 ([mattlanders.net][1])
3. 通过一个极小的 MDP（Baird 反例变体）**亲自跑出参数爆炸**，并知道每一行代码在模拟什么、为什么这么做。

---

## 1) 三件事分别是什么（严格口径）

我们先把“致命三元组”写成可操作的三条：

1. **函数逼近（Function Approximation）**
   不是用表格存 (V(s))，而是用参数 (\theta) 表示：
   [
   \hat V(s;\theta)=\phi(s)^\top \theta
   ]
   这意味着**不同状态共享参数**，一次更新会“连带”影响很多状态（泛化）。

2. **自举（Bootstrapping）**
   TD(0) 的目标不是完整回报 (G_t)，而是用下一步的估计值做目标：
   [
   \text{target}=r_{t+1}+\gamma \hat V(s_{t+1};\theta)
   ]
   目标里包含当前网络/参数的输出，**目标会跟着参数变化**（这就是“自举”）。

3. **离策略（Off-policy）**
   你用行为策略 (b) 采样数据，但想评估/优化目标策略 (\pi)。
   常用做法是重要性采样比率：
   [
   \rho_t=\frac{\pi(a_t|s_t)}{b(a_t|s_t)}
   ]

把三者合在一起（线性函数逼近 + TD 自举 + off-policy/重要性采样），使用常见的半梯度 TD 更新：
[
\theta \leftarrow \theta + \alpha\ \rho_t\ \delta_t\ \phi(s_t),\quad
\delta_t=r_{t+1}+\gamma \hat V(s_{t+1};\theta)-\hat V(s_t;\theta)
]
就可能出现：(|\theta|) 越学越大，直接爆炸（发散）。 ([mattlanders.net][1])

---

## 2) 直觉：为什么“三者一起”会出事（只讲这一点，讲透）

核心直觉只有一句话：

> **更新方向不再等价于“在某个合理损失上做下降”**，而共享参数的“连带效应”会把误差在状态之间“泵大”，自举又把这个泵大后的估计继续当作目标，off-policy 的分布偏差让这种“泵大”得不到及时纠正，于是形成正反馈回路。

在 Baird 反例里，行为策略为了覆盖更多状态，经常走某条分支；但目标策略实际上几乎不走那条分支。结果就是：

* 你**频繁访问**很多“上层状态”（来自行为策略），这些状态的更新会把某些共享参数持续往一个方向推；
* 真正能把参数“拉回去”的校正更新（与目标策略一致的那部分）**发生得少**，甚至带很大 (\rho)（高方差、更新不稳定）；
* 于是出现“增长 > 校正”的长期净效应，参数不断被推大，最终爆炸。 ([mattlanders.net][1])

[1]: https://mattlanders.net/the-deadly-triad.html "The Deadly Triad"
---

# 第 10 课 10.5：投影贝尔曼方程（Projected Bellman Equation）——为什么 Semi-gradient TD 不是在最小化“价值误差”

你上一节看到了“致命三元组会发散”。这一节我们把原因从直觉提升到结构层面，只讲一个点：**在函数逼近下，TD(0) 收敛到的不是“最小化真实价值误差”的解，而是“投影贝尔曼方程”的不动点**。这也是为什么它在某些条件下稳定、在另一些条件下会出事的根。

---

1. 清楚区分两个“看起来很像、但本质不同”的目标：

   * **最小化价值误差（MSVE）**：让 (\hat v) 尽量逼近真实 (v^\pi)
   * **满足投影贝尔曼不动点（PBE）**：(\hat v = \Pi T^\pi \hat v)
2. 知道 TD(0)（线性函数逼近、on-policy）收敛到的 (\theta_{\text{TD}}) 满足一个线性方程 (A\theta=b)（这是 PBE 的代数形式）。
3. 通过一个 3 状态小例子，亲眼看到：
   [
   \theta_{\text{TD}} \neq \theta_{\text{LS}}
   ]
   且 (\theta_{\text{LS}}) 是“最小 MSVE 的解”，(\theta_{\text{TD}}) 是“PBE 不动点”。

---

### 1) 两个对象：真实贝尔曼方程 vs 投影贝尔曼方程

对固定策略 (\pi)（或 MRP），定义贝尔曼算子：
[
(T^\pi v)(s) ;=; r(s) + \gamma \sum_{s'} P(s'|s),v(s')
]

真实价值函数满足：
[
v^\pi = T^\pi v^\pi
]

但函数逼近把价值限制在一个子空间里：
[
\hat v_\theta = \Phi \theta
]
（(\Phi) 是特征矩阵，每行是 (\phi(s)^\top)）

通常 (v^\pi) **不在**这个子空间里，所以你不可能让 (\Phi\theta = v^\pi) 精确成立。

于是就有两种“合理的”追求：

#### 目标 A：最小化“逼近误差”（MSVE / Least Squares）

用某个状态分布 (d)（on-policy 常用平稳分布）定义加权平方误差：
[
\text{MSVE}(\theta)=|\Phi\theta - v^\pi|*D^2,\quad D=\mathrm{diag}(d)
]
最小化它得到：
[
\theta*{\text{LS}}=\arg\min_\theta |\Phi\theta - v^\pi|_D^2
]

#### 目标 B：满足“投影贝尔曼方程”（PBE）

把贝尔曼更新 (T^\pi(\Phi\theta)) 投影回可表示子空间（投影算子 (\Pi) 是对 (D)-内积的正交投影）：
[
\Phi\theta ;=; \Pi, T^\pi(\Phi\theta)
]
这就是 **Projected Bellman Equation**。

**TD(0)（线性、on-policy）收敛到的就是这个方程的不动点**，即：
[
\theta_{\text{TD}}:\ \Phi\theta_{\text{TD}} = \Pi T^\pi(\Phi\theta_{\text{TD}})
]

---

### 2) TD(0) 为什么不是在最小化 MSVE（关键结论）

如果 TD(0) 真的是对 MSVE 做梯度下降，它应当收敛到 (\theta_{\text{LS}})。
但事实是：线性 TD(0) 收敛到 (\theta_{\text{TD}})，一般 **不等于** (\theta_{\text{LS}})。

更具体地，线性 TD(0) 的固定点满足一个线性系统：
[
A\theta=b
]
其中在 on-policy 情况（MRP）常写为：
[
A = \Phi^\top D (I-\gamma P)\Phi,\qquad b = \Phi^\top D r
]
这对应的就是 PBE 的代数形式。

而最小 MSVE 的解满足：
[
(\Phi^\top D \Phi)\theta = \Phi^\top D v^\pi
]
两个方程一般不相同，因此解也一般不相同。

这就是你要牢牢记住的：
**TD(0) 的“目标”不是逼近真实 (v^\pi) 的最小二乘解，而是投影贝尔曼不动点。**

---

# 第 10 课 10.6：MSPBE 与 GTD2 / TDC——把“TD 固定点问题”变成“真正的随机梯度下降”，从而在 off-policy 下恢复收敛

上一节你已经明确：**线性 TD(0)（semi-gradient）收敛的是投影贝尔曼不动点（PBE），而不是最小 MSVE**。这一节我们只讲一个点，但讲深：**MSPBE（Mean Squared Projected Bellman Error）是什么，以及 GTD2 / TDC 为什么能在“函数逼近 + 自举 + off-policy”里保持稳定（至少在线性设定下）。**

---

1. 写出 MSPBE 的定义，并理解它为何是“可优化的目标函数”。
2. 看到关键代数形式：
   [
   \mathrm{MSPBE}(\theta)=(b-A\theta)^\top C^{-1}(b-A\theta)
   ]
   并解释 (A,b,C) 各自代表什么、来自哪里。
3. 理解 GTD2 / TDC 的核心结构：**引入辅助向量 (w)** 来近似 (C^{-1}(b-A\theta))，从而实现对 MSPBE 的随机梯度下降。
4. 跑出一个对照实验：在 Baird 反例上

   * off-policy 线性 TD(0) **参数爆炸**
   * GTD2 / TDC **参数有界并趋向稳定**（MSPBE 估计下降或保持很小）

---

## 1) MSPBE：把“投影贝尔曼误差”当成真正的优化目标

在函数逼近下 (\hat v_\theta=\Phi\theta)，贝尔曼算子是 (T^\pi)。我们关心的是“投影后”的误差：
[
\Pi T^\pi(\Phi\theta) - \Phi\theta
]
MSPBE 定义为其在 (D)-加权范数下的平方：
[
\mathrm{MSPBE}(\theta)=\left|\Pi T^\pi(\Phi\theta) - \Phi\theta \right|_D^2
]

关键点：**MSPBE 是一个标量目标函数**，你可以谈“梯度”“下降”，这与上一节的 semi-gradient TD 形成对比：semi-gradient TD 的更新一般不是某个简单标量目标的真实梯度下降。

---

## 2) 线性情形下，MSPBE 可以写成一个非常实用的二次型

在线性设定（MRP / 固定策略）下，定义：

* 特征：(\phi_t=\phi(s_t))，(\phi_{t+1}=\phi(s_{t+1}))
* 重要性比率：(\rho_t=\frac{\pi(a_t|s_t)}{b(a_t|s_t)})（off-policy 才需要）
* TD 误差：(\delta_t=r_{t+1}+\gamma \theta^\top \phi_{t+1}-\theta^\top \phi_t)

然后定义三个“矩阵/向量的期望”（这是最核心的对象）：

[
A = \mathbb{E}\left[ \rho_t , \phi_t(\phi_t-\gamma\phi_{t+1})^\top \right]
]
[
b = \mathbb{E}\left[ \rho_t , r_{t+1}, \phi_t \right]
]
[
C = \mathbb{E}\left[ \phi_t \phi_t^\top \right]
]

则 MSPBE 可写成：
[
\boxed{\mathrm{MSPBE}(\theta)=(b-A\theta)^\top C^{-1}(b-A\theta)}
]

解释一下这三个对象在做什么（你需要能讲出来）：

* **(b-A\theta)**：是“投影贝尔曼方程的残差”（PBE 残差）在特征空间里的表达
* **(C)**：是特征协方差（由行为分布决定），(C^{-1}) 对残差做了一个“按特征相关性去耦”的度量
* 这个形式说明 MSPBE 是一个二次函数（线性情形），非常适合做梯度法

---

## 3) GTD2 / TDC：为什么要引入辅助变量 (w)

对二次型求梯度可以得到（你只需记结构）：
[
\nabla_\theta \mathrm{MSPBE}(\theta) = -2 A^\top C^{-1}(b-A\theta)
]

困难在于：你无法直接拿到 (C^{-1})（也不想每步矩阵求逆）。

GTD 的技巧：引入一个辅助向量 (w)，让它去逼近：
[
w \approx C^{-1}(b-A\theta)
]
即让 (w) 近似解线性方程：
[
Cw = (b-A\theta)
]

一旦你有了 (w)，梯度方向就近似为：
[
A^\top w
]
并且 (A^\top w) 可以被构造成**单样本无偏估计**，从而得到可实现的随机梯度下降更新。

这就是 GTD2 / TDC 的本质：

> **用两时间尺度（(\beta>\alpha)）让 (w) 快速跟踪 (C^{-1}(b-A\theta))，再用它更新 (\theta)。**

---
