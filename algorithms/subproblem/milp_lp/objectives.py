# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
from .problem_data import ProblemData


def resource_usage(problem: ProblemData, y: np.ndarray) -> np.ndarray:
    """
    资源使用量：
        A y
    """
    return problem.consumption_matrix @ y


def remaining_resource(problem: ProblemData, y: np.ndarray) -> np.ndarray:
    """
    剩余资源：
        R - A y
    """
    return problem.resource_limit - resource_usage(problem, y)


def lp_objective_value(problem: ProblemData, y: np.ndarray) -> float:
    """
    固定 x 后，连续子问题目标值（不含固定成本）：
        sum_i (p_i - alpha_i) y_i
    """
    return float(np.sum(problem.c * y))


def total_objective_value(problem: ProblemData, x: np.ndarray, y: np.ndarray) -> float:
    """
    原问题总目标值：
        sum_i [ (p_i - alpha_i) y_i - f_i x_i ]
    """
    return float(np.sum(problem.c * y - problem.fixed_cost * x))