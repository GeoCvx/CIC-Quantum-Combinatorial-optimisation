# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class QUBOConfig:
    """
    QUBO 构造参数。

    objective_scale:
        目标项缩放系数。默认 1.0。

    resource_penalty:
        资源约束罚项系数。越大越强调可行性。

    hamming_penalty:
        与 incumbent 的 Hamming 距离正则项系数。
        设为 0 表示不启用。

    demand_weight:
        对 max_demand 的启发式利用强度。
        由于主问题只优化 x，不直接优化 y，
        这里用 max_demand 近似“若产品被选中，其潜在贡献规模”。
    """
    objective_scale: float = 1.0
    resource_penalty: float = 10.0
    hamming_penalty: float = 0.0
    demand_weight: float = 1.0


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


def _extract_arrays(problem_dict: dict):
    n = int(problem_dict["product_count"])
    m = int(problem_dict["resource_count"])

    price = np.asarray(problem_dict["price"], dtype=float).reshape(n)
    fixed_cost = np.asarray(problem_dict["fixed_cost"], dtype=float).reshape(n)
    alpha = np.asarray(problem_dict["alpha"], dtype=float).reshape(n)
    max_demand = np.asarray(problem_dict["max_demand"], dtype=float).reshape(n)
    resource_limit = np.asarray(problem_dict["resource_limit"], dtype=float).reshape(m)
    A = np.asarray(problem_dict["consumption_matrix"], dtype=float).reshape(m, n)

    return n, m, price, fixed_cost, alpha, max_demand, resource_limit, A


def _build_profit_linear_term(
    price: np.ndarray,
    alpha: np.ndarray,
    fixed_cost: np.ndarray,
    max_demand: np.ndarray,
    demand_weight: float,
    objective_scale: float,
) -> np.ndarray:
    """
    构造主问题中 x 的线性收益近似项。

    由于真实问题中 x 的收益要通过子问题 y 才能准确评估，
    这里用启发式近似：
        estimated_profit_i = (price_i - alpha_i) * demand_weight * max_demand_i - fixed_cost_i

    为了写成最小化 QUBO:
        min x^T Q x
    我们把“希望选取高收益产品”改写为负线性项。
    """
    unit_margin = price - alpha
    estimated_profit = unit_margin * demand_weight * max_demand - fixed_cost

    # QUBO 是最小化，所以利润越高，对角项越负
    linear_diag = -objective_scale * estimated_profit
    return linear_diag


def _add_resource_penalty(
    Q: np.ndarray,
    A: np.ndarray,
    max_demand: np.ndarray,
    resource_limit: np.ndarray,
    penalty: float,
) -> None:
    """
    添加资源约束的二次罚项。

    对每个资源 j，考虑近似约束：
        sum_i a_{j,i} * max_demand_i * x_i <= R_j

    使用平方罚：
        penalty * (sum_i w_i x_i - R_j)^2

    展开后：
        penalty * [x^T (w w^T) x - 2 R_j w^T x + 常数]

    常数项对优化无影响，不写入 Q。
    """
    m, n = A.shape
    for j in range(m):
        w = A[j] * max_demand
        Rj = float(resource_limit[j])

        # 二次项
        Q += penalty * np.outer(w, w)

        # 线性项写入对角线
        Q[np.arange(n), np.arange(n)] += -2.0 * penalty * Rj * w


def _add_hamming_regularization(
    Q: np.ndarray,
    incumbent: np.ndarray,
    penalty: float,
) -> None:
    """
    添加与 incumbent 的 Hamming 正则项：

        penalty * sum_i (x_i - x*_i)^2

    因为 x_i ∈ {0,1}，有 x_i^2 = x_i，
    展开可得：
        penalty * sum_i [ (1 - 2 x*_i) x_i ] + const

    常数忽略，只写入对角线。
    """
    incumbent = np.asarray(incumbent, dtype=float).reshape(-1)
    if np.any((incumbent != 0.0) & (incumbent != 1.0)):
        raise ValueError("incumbent 必须是 0/1 向量。")

    linear_diag = penalty * (1.0 - 2.0 * incumbent)
    Q[np.arange(len(incumbent)), np.arange(len(incumbent))] += linear_diag


def symmetrize_qubo(Q: np.ndarray) -> np.ndarray:
    """
    保证 Q 为对称矩阵。
    """
    return 0.5 * (Q + Q.T)


def build_qubo(
    problem_dict: dict,
    config: QUBOConfig | None = None,
    incumbent: np.ndarray | None = None,
) -> np.ndarray:
    """
    构造主问题近似 QUBO：
        min x^T Q x
        s.t. x ∈ {0,1}^n

    返回
    ----
    Q : np.ndarray, shape (n, n)
        QUBO 矩阵
    """
    if config is None:
        config = QUBOConfig()

    _validate_problem_dict(problem_dict)
    n, m, price, fixed_cost, alpha, max_demand, resource_limit, A = _extract_arrays(problem_dict)

    Q = np.zeros((n, n), dtype=float)

    # 1) 目标项（线性收益近似）
    linear_diag = _build_profit_linear_term(
        price=price,
        alpha=alpha,
        fixed_cost=fixed_cost,
        max_demand=max_demand,
        demand_weight=config.demand_weight,
        objective_scale=config.objective_scale,
    )
    Q[np.arange(n), np.arange(n)] += linear_diag

    # 2) 资源罚项
    _add_resource_penalty(
        Q=Q,
        A=A,
        max_demand=max_demand,
        resource_limit=resource_limit,
        penalty=config.resource_penalty,
    )

    # 3) 与 incumbent 的 Hamming 正则
    if incumbent is not None and config.hamming_penalty > 0.0:
        incumbent = np.asarray(incumbent, dtype=float).reshape(n)
        _add_hamming_regularization(
            Q=Q,
            incumbent=incumbent,
            penalty=config.hamming_penalty,
        )

    return symmetrize_qubo(Q)


def qubo_objective_value(Q: np.ndarray, x: np.ndarray) -> float:
    """
    计算 QUBO 目标值：
        x^T Q x
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    return float(x @ Q @ x)


def random_binary_x(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    生成随机 0/1 向量。
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, 2, size=n).astype(float)