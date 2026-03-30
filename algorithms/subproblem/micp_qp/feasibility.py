# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
from .problem_data import ProblemData
from .objectives import resource_usage


def validate_x_binary(x: np.ndarray, n: int, tol: float = 1e-9) -> np.ndarray:
    """
    检查 x 是否为 0/1 向量。
    若元素数值上接近 0/1，则自动四舍五入。
    """
    x = np.asarray(x, dtype=float).reshape(-1)

    if x.shape != (n,):
        raise ValueError(f"x shape 应为 ({n},)，实际为 {x.shape}")

    rounded = np.round(x)
    if np.any(np.abs(x - rounded) > tol):
        raise ValueError("x 不是合法的 0/1 向量。")

    rounded = rounded.astype(float)
    if np.any((rounded != 0.0) & (rounded != 1.0)):
        raise ValueError("x 中元素必须为 0 或 1。")

    return rounded


def upper_bound_from_x(problem: ProblemData, x: np.ndarray) -> np.ndarray:
    """
    固定 x 后，y 的上界：
        0 <= y_i <= D_i x_i
    """
    return problem.max_demand * x


def check_primal_feasibility(
    problem: ProblemData,
    x: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-7,
) -> dict:
    """
    检查原始可行性：
        A y <= R
        0 <= y <= D * x
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    u = upper_bound_from_x(problem, x)
    usage = resource_usage(problem, y)

    lower_violation = max(0.0, float(np.max(-y)))
    upper_violation = max(0.0, float(np.max(y - u)))
    resource_violation = max(0.0, float(np.max(usage - problem.resource_limit)))

    feasible = (
        lower_violation <= tol
        and upper_violation <= tol
        and resource_violation <= tol
    )

    return {
        "feasible": feasible,
        "lower_violation": lower_violation,
        "upper_violation": upper_violation,
        "resource_violation": resource_violation,
    }


def repair_y_numerically(
    problem: ProblemData,
    x: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    对数值误差造成的轻微越界做修复：
    1) clip 到 [0, D*x]
    2) 若 A y > R，则统一缩放
    """
    y = np.asarray(y, dtype=float).copy()
    u = upper_bound_from_x(problem, x)

    # 先修盒约束
    y = np.clip(y, 0.0, u)

    usage = problem.consumption_matrix @ y
    max_over = max(0.0, float(np.max(usage - problem.resource_limit)))

    if max_over <= tol:
        return y

    # 若资源约束轻微超出，做保守缩放
    mask = usage > tol
    if np.any(mask):
        ratios = problem.resource_limit[mask] / usage[mask]
        gamma = min(1.0, float(np.min(ratios)))
        gamma = max(0.0, gamma)
        y = gamma * y
        y = np.clip(y, 0.0, u)

    return y