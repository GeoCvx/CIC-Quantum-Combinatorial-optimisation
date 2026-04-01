# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from algorithms.preprocess.feature_builder import ProblemStats, build_problem_stats
from algorithms.feedback.bias_updater import BiasState


@dataclass
class QUBOConfig:
    objective_scale: float = 1.0
    resource_penalty: float = 10.0
    hamming_penalty: float = 0.0
    demand_weight: float = 1.0
    conflict_weight: float = 1.0

    # 新增：tabu / elite / stagnation
    tabu_penalty: float = 5.0
    elite_weight: float = 0.5
    exploration_boost: float = 0.5


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
    fixed_cost = np.asarray(problem_dict["fixed_cost"], dtype=float).reshape(n)
    max_demand = np.asarray(problem_dict["max_demand"], dtype=float).reshape(n)
    resource_limit = np.asarray(problem_dict["resource_limit"], dtype=float).reshape(m)
    A = np.asarray(problem_dict["consumption_matrix"], dtype=float).reshape(m, n)
    return n, m, fixed_cost, max_demand, resource_limit, A


def _add_resource_penalty(
    Q: np.ndarray,
    A: np.ndarray,
    approx_output: np.ndarray,
    resource_limit: np.ndarray,
    penalty: float,
) -> None:
    m, n = A.shape
    diag_idx = np.arange(n)

    for j in range(m):
        w = A[j] * approx_output
        Rj = float(resource_limit[j])
        Q += penalty * np.outer(w, w)
        Q[diag_idx, diag_idx] += -2.0 * penalty * Rj * w


def _add_hamming_regularization(
    Q: np.ndarray,
    incumbent: np.ndarray,
    penalty: float,
) -> None:
    incumbent = np.asarray(incumbent, dtype=float).reshape(-1)
    diag_idx = np.arange(len(incumbent))
    # x_i != incumbent_i 时增加代价
    Q[diag_idx, diag_idx] += penalty * (1.0 - 2.0 * incumbent)


def _add_tabu_penalty(
    Q: np.ndarray,
    tabu_set: list[str],
    penalty: float,
) -> None:
    if not tabu_set:
        return

    n = Q.shape[0]
    diag_idx = np.arange(n)

    for bitstring in tabu_set:
        x = np.array([int(ch) for ch in bitstring[:n]], dtype=float)

        # 单体惩罚
        Q[diag_idx, diag_idx] += penalty * x

        # 组合惩罚：共现的 1-1 对一起加压
        active = np.where(x > 0.5)[0]
        for idx_i in range(len(active)):
            for idx_j in range(idx_i + 1, len(active)):
                i = active[idx_i]
                j = active[idx_j]
                Q[i, j] += penalty
                Q[j, i] += penalty


def _add_elite_attraction(
    Q: np.ndarray,
    elite_pool: list[dict],
    weight: float,
) -> None:
    if not elite_pool:
        return

    n = Q.shape[0]
    diag_idx = np.arange(n)

    freq = np.zeros(n, dtype=float)
    for item in elite_pool:
        freq += np.asarray(item["x"], dtype=float).reshape(-1)
    freq /= max(1, len(elite_pool))

    # 高频出现在 elite 中的变量，在下一轮更容易被选中
    Q[diag_idx, diag_idx] += -weight * freq


def _add_stagnation_exploration(
    Q: np.ndarray,
    no_improve_rounds: int,
    strength: float,
    seed: int = 42,
) -> None:
    if no_improve_rounds <= 0 or strength <= 0.0:
        return

    rng = np.random.default_rng(seed + int(no_improve_rounds))
    n = Q.shape[0]
    noise = rng.uniform(-1.0, 1.0, size=n)
    Q[np.arange(n), np.arange(n)] += strength * no_improve_rounds * noise


def symmetrize_qubo(Q: np.ndarray) -> np.ndarray:
    return 0.5 * (Q + Q.T)


def build_qubo(
    problem_dict: dict,
    config: QUBOConfig | None = None,
    incumbent: np.ndarray | None = None,
    stats: ProblemStats | None = None,
    bias_state: BiasState | None = None,
) -> np.ndarray:
    if config is None:
        config = QUBOConfig()

    _validate_problem_dict(problem_dict)

    if stats is None:
        stats = build_problem_stats(problem_dict)

    n, m, fixed_cost, max_demand, resource_limit, A = _extract_arrays(problem_dict)
    Q = np.zeros((n, n), dtype=float)
    diag_idx = np.arange(n)

    # 1) 单体收益
    linear_profit = stats.single_profit.copy()
    if bias_state is not None:
        linear_profit = linear_profit + bias_state.linear_bias
    Q[diag_idx, diag_idx] += -config.objective_scale * linear_profit

    # 2) 产品冲突
    conflict_matrix = stats.conflict_matrix.copy()
    if bias_state is not None:
        conflict_matrix = conflict_matrix + bias_state.pair_bias
    Q += config.conflict_weight * conflict_matrix

    # 3) 资源近似罚项
    effective_penalty = config.resource_penalty
    if bias_state is not None:
        effective_penalty *= bias_state.feasibility_penalty_scale

    _add_resource_penalty(
        Q=Q,
        A=A,
        approx_output=stats.approx_output,
        resource_limit=resource_limit,
        penalty=effective_penalty,
    )

    # 4) incumbent 正则
    reg_penalty = config.hamming_penalty
    if bias_state is not None:
        reg_penalty *= bias_state.regularization_scale
    if incumbent is not None and reg_penalty > 0.0:
        _add_hamming_regularization(
            Q,
            incumbent=np.asarray(incumbent, dtype=float).reshape(n),
            penalty=reg_penalty,
        )

    # 5) tabu / no-good
    if bias_state is not None and bias_state.tabu_set:
        _add_tabu_penalty(Q, bias_state.tabu_set, penalty=config.tabu_penalty)

    # 6) elite 结构吸引
    if bias_state is not None and bias_state.elite_pool:
        _add_elite_attraction(Q, bias_state.elite_pool, weight=config.elite_weight)

    # 7) 停滞时增强探索
    if bias_state is not None and bias_state.no_improve_rounds > 0:
        _add_stagnation_exploration(
            Q,
            no_improve_rounds=bias_state.no_improve_rounds,
            strength=config.exploration_boost,
        )

    return symmetrize_qubo(Q)


def qubo_objective_value(Q: np.ndarray, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    return float(x @ Q @ x)


def random_binary_x(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, 2, size=n).astype(float)