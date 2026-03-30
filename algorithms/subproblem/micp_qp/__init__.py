# -*- coding: utf-8 -*-
"""
qp_solver package

固定 x 后的 QP 子问题求解器：
    max_y  sum_i [ (p_i - alpha_i) y_i - beta_i y_i^2 ]
    s.t.   A y <= R
           0 <= y_i <= D_i x_i

当前版本：
- 仅支持 beta_i > 0
- 使用拉格朗日对偶 + 对偶空间投影梯度
"""

# -*- coding: utf-8 -*-

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