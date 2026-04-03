# -*- coding: utf-8 -*-
"""
qp_solver package

固定 x 后的 QP / convex subproblem 求解器：
    max_y  sum_i [ (p_i - alpha_i) y_i - beta_i y_i^2 ]
    s.t.   A y <= R
           0 <= y_i <= D_i x_i

当前版本：
- 支持 beta 不全为 0 的情形
- 允许存在部分 beta_i = 0
- 若 beta 全为 0，应交给 LP 求解器处理
- 当存在 beta_i = 0 时，内部自动切换为“投影次梯度法”
"""

from .problem_data import ProblemData
from .dual_solver import QPSolverConfig, QPSolverResult, solve_qp_given_x
from .evaluate import evaluate_x

__all__ = [
    "ProblemData",
    "QPSolverConfig",
    "QPSolverResult",
    "solve_qp_given_x",
    "evaluate_x",
]