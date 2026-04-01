# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ProblemStats:
    n: int
    m: int
    approx_output: np.ndarray
    single_profit: np.ndarray
    resource_pressure: np.ndarray
    conflict_matrix: np.ndarray
    unit_margin: np.ndarray


def _validate_problem_dict(problem_dict: dict) -> None:
    required = [
        "product_count",
        "resource_count",
        "price",
        "fixed_cost",
        "alpha",
        "max_demand",
        "resource_limit",
        "consumption_matrix",
    ]
    for key in required:
        if key not in problem_dict:
            raise KeyError(f"problem_dict 缺少字段: {key}")


def build_problem_stats(problem_dict: dict) -> ProblemStats:
    """
    统一预处理入口。

    产出：
    - approx_output: 单产品近似产量
    - single_profit: 与问题类型一致的单产品近似收益
    - resource_pressure: 资源压力指标
    - conflict_matrix: 产品对冲突矩阵
    """
    _validate_problem_dict(problem_dict)

    n = int(problem_dict["product_count"])
    m = int(problem_dict["resource_count"])

    price = np.asarray(problem_dict["price"], dtype=float).reshape(n)
    fixed_cost = np.asarray(problem_dict["fixed_cost"], dtype=float).reshape(n)
    alpha = np.asarray(problem_dict["alpha"], dtype=float).reshape(n)
    beta = np.asarray(problem_dict["beta"], dtype=float).reshape(n)
    demand = np.asarray(problem_dict["max_demand"], dtype=float).reshape(n)
    resource_limit = np.asarray(problem_dict["resource_limit"], dtype=float).reshape(m)
    A = np.asarray(problem_dict["consumption_matrix"], dtype=float).reshape(m, n)

    approx_output = np.zeros(n, dtype=float)
    for i in range(n):
        positive_rows = A[:, i] > 1e-12
        if not np.any(positive_rows):
            approx_output[i] = demand[i]
        else:
            upper_by_resource = resource_limit[positive_rows] / A[positive_rows, i]
            approx_output[i] = min(float(demand[i]), float(np.min(upper_by_resource)))

    unit_margin = price - alpha
    single_profit = unit_margin * approx_output - fixed_cost

    # MICP 情况下，用凹二次项的近似收益做单体统计量
    has_quadratic = bool(np.any(np.abs(beta) > 1e-12))
    if has_quadratic:
        single_profit = unit_margin * approx_output - beta * approx_output * approx_output - fixed_cost

    resource_pressure = np.sum(A / np.maximum(resource_limit.reshape(-1, 1), 1e-12), axis=0)

    # 共享资源上的冲突强度
    conflict_matrix = np.zeros((n, n), dtype=float)
    scaled_A = A / np.maximum(resource_limit.reshape(-1, 1), 1e-12)
    for i in range(n):
        for j in range(i + 1, n):
            c = float(np.dot(scaled_A[:, i], scaled_A[:, j]))
            conflict_matrix[i, j] = c
            conflict_matrix[j, i] = c

    return ProblemStats(
        n=n,
        m=m,
        approx_output=approx_output,
        single_profit=single_profit,
        resource_pressure=resource_pressure,
        conflict_matrix=conflict_matrix,
        unit_margin=unit_margin,
    )
