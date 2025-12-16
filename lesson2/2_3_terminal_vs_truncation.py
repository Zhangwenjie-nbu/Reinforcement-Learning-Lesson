# -*- coding: utf-8 -*-
"""
lesson2_3_terminal_vs_truncation.py

整体在做什么？
1) 构造一个极简的“持续任务”(continuing task)：每一步奖励恒为 +1，理论上永不自然终止。
2) 引入工程常见的 time-limit（最多 H 步），形成“截断 truncated”，但这并不等价于“自然终止 terminal”。
3) 对比两种回报/价值的计算：
   - 理论上的无限时域折扣回报：V = 1/(1-gamma)
   - 把 time-limit 截断当作 terminal（未来价值=0）时得到的有限回报
4) 输出两者差异，直观看到“把截断当终止”会系统性低估价值。

你需要从这份代码理解：
- terminated vs truncated 的区别为什么重要
- 为什么很多 RL 实现会单独处理 time-limit 截断，而不是简单 done=1
"""

def true_infinite_discounted_value(gamma: float) -> float:
    """
    计算 continuing task（无限步、每步奖励为1）在折扣 gamma 下的理论真值：
    V = 1 + gamma + gamma^2 + ... = 1/(1-gamma)

    逻辑：
    - 这是一个等比数列求和，用于给出“正确答案”，作为对照基准。
    """
    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must be in [0, 1).")
    return 1.0 / (1.0 - gamma)


def truncated_discounted_return(gamma: float, horizon: int) -> float:
    """
    计算“最多执行 horizon 步就截断”时的折扣回报（把截断当作结束）：
    G = 1 + gamma + ... + gamma^(horizon-1)

    逻辑：
    - 这等价于把“第 horizon 步之后的所有未来回报”强行当作 0。
    - 这在 continuing task 中会带来系统性低估。
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    G = 0.0
    power = 1.0
    for _ in range(horizon):
        G += power * 1.0  # 每步奖励恒为1
        power *= gamma
    return G


def bias_from_treating_truncation_as_terminal(gamma: float, horizon: int) -> float:
    """
    计算“把 time-limit 截断当 terminal”带来的价值低估偏差：
    bias = V_true_infinite - G_truncated

    逻辑：
    - continuing task 的真值是无限和
    - time-limit 截断得到的是有限和
    - 差值就是由于错误终止假设引入的偏差
    """
    v_true = true_infinite_discounted_value(gamma)
    g_trunc = truncated_discounted_return(gamma, horizon)
    return v_true - g_trunc


if __name__ == "__main__":
    gamma = 0.95
    horizons = [5, 10, 50, 200]

    v_true = true_infinite_discounted_value(gamma)
    print(f"gamma = {gamma}")
    print(f"True infinite-horizon value V = 1/(1-gamma) = {v_true:.6f}\n")

    for H in horizons:
        g_trunc = truncated_discounted_return(gamma, H)
        bias = bias_from_treating_truncation_as_terminal(gamma, H)
        print(f"Horizon H = {H:>3} | truncated return = {g_trunc:.6f} | bias (underestimate) = {bias:.6f}")
