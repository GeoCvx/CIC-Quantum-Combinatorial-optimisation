# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from algorithms.preprocess.feature_builder import ProblemStats


@dataclass
class BiasUpdateConfig:
    reward_step: float = 0.10
    penalty_step: float = 0.20
    pair_penalty_step: float = 0.15

    # elite 高频结构轻量强化
    elite_reward_step: float = 0.05

    # 可行但差的候选，若目标低于 best * feasible_bad_ratio，则进入 no-good 逻辑
    feasible_bad_ratio: float = 0.98

    max_abs_linear_bias: float = 8.0
    max_abs_pair_bias: float = 8.0

    infeasible_penalty_growth: float = 1.10

    # 停滞时：
    # - regularization_scale 下降，避免总被 incumbent 吸住
    # - feasibility_penalty_scale 上升，增强结构变化
    no_improve_regularization_decay: float = 0.90
    no_improve_feasibility_growth: float = 1.05

    max_tabu_size: int = 50
    elite_max_size: int = 10


@dataclass
class BiasState:
    linear_bias: np.ndarray
    pair_bias: np.ndarray
    feasibility_penalty_scale: float = 1.0
    regularization_scale: float = 1.0
    incumbent: np.ndarray | None = None
    elite_pool: list[dict[str, Any]] = field(default_factory=list)
    tabu_set: list[str] = field(default_factory=list)
    no_improve_rounds: int = 0


def init_bias_state(stats: ProblemStats) -> BiasState:
    return BiasState(
        linear_bias=np.zeros(stats.n, dtype=float),
        pair_bias=np.zeros((stats.n, stats.n), dtype=float),
    )


def _bitstring_from_x(x: np.ndarray) -> str:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return "".join(str(int(round(v))) for v in arr)


def _clip_bias(state: BiasState, cfg: BiasUpdateConfig) -> None:
    state.linear_bias = np.clip(
        state.linear_bias,
        -cfg.max_abs_linear_bias,
        cfg.max_abs_linear_bias,
    )
    state.pair_bias = np.clip(
        state.pair_bias,
        -cfg.max_abs_pair_bias,
        cfg.max_abs_pair_bias,
    )
    state.pair_bias = 0.5 * (state.pair_bias + state.pair_bias.T)
    np.fill_diagonal(state.pair_bias, 0.0)

    state.feasibility_penalty_scale = max(0.5, min(10.0, state.feasibility_penalty_scale))
    state.regularization_scale = max(0.2, min(5.0, state.regularization_scale))


def _update_elite_pool(
    state: BiasState,
    candidate_records: list[dict[str, Any]],
    cfg: BiasUpdateConfig,
) -> None:
    state.elite_pool.extend(candidate_records)
    state.elite_pool.sort(key=lambda r: float(r["objective_value"]), reverse=True)

    deduped: list[dict[str, Any]] = []
    seen = set()
    for rec in state.elite_pool:
        key = _bitstring_from_x(rec["x"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)
        if len(deduped) >= cfg.elite_max_size:
            break
    state.elite_pool = deduped


def _push_tabu(state: BiasState, x: np.ndarray, cfg: BiasUpdateConfig) -> None:
    key = _bitstring_from_x(x)
    if key not in state.tabu_set:
        state.tabu_set.append(key)
    if len(state.tabu_set) > cfg.max_tabu_size:
        state.tabu_set = state.tabu_set[-cfg.max_tabu_size:]


def _add_pair_penalty(pair_bias: np.ndarray, x: np.ndarray, step: float) -> None:
    active = np.where(np.asarray(x, dtype=float).reshape(-1) > 0.5)[0]
    for idx_i in range(len(active)):
        for idx_j in range(idx_i + 1, len(active)):
            i = active[idx_i]
            j = active[idx_j]
            pair_bias[i, j] += step
            pair_bias[j, i] += step


def update_bias_state(
    bias_state: BiasState,
    evaluated_candidates: list[dict[str, Any]],
    previous_best_objective: float,
    stats: ProblemStats,
    config: BiasUpdateConfig | None = None,
) -> BiasState:
    if config is None:
        config = BiasUpdateConfig()

    if not evaluated_candidates:
        bias_state.no_improve_rounds += 1
        bias_state.regularization_scale *= config.no_improve_regularization_decay
        bias_state.feasibility_penalty_scale *= config.no_improve_feasibility_growth
        _clip_bias(bias_state, config)
        return bias_state

    # 以真实目标排序
    evaluated_sorted = sorted(
        evaluated_candidates,
        key=lambda r: float(r["objective_value"]),
        reverse=True,
    )
    best_record = evaluated_sorted[0]
    best_x = np.asarray(best_record["x"], dtype=float).reshape(-1)

    improved = float(best_record["objective_value"]) > float(previous_best_objective)

    if improved:
        bias_state.no_improve_rounds = 0
        bias_state.incumbent = best_x.copy()
    else:
        bias_state.no_improve_rounds += 1
        # 停滞：减弱 incumbent 吸附，增强探索
        bias_state.regularization_scale *= config.no_improve_regularization_decay
        bias_state.feasibility_penalty_scale *= config.no_improve_feasibility_growth

    feasible_records = [r for r in evaluated_candidates if bool(r.get("feasible", True))]
    infeasible_records = [r for r in evaluated_candidates if not bool(r.get("feasible", True))]

    # 1) 奖励高质量可行候选中的单体
    top_feasible = feasible_records[: max(1, min(3, len(feasible_records)))]
    for rec in top_feasible:
        x = np.asarray(rec["x"], dtype=float).reshape(-1)
        bias_state.linear_bias += config.reward_step * x

    # 2) 惩罚不可行候选：线性 + pair + 提高可行性罚项 + 进入 tabu
    for rec in infeasible_records[: max(1, min(3, len(infeasible_records)))]:
        x = np.asarray(rec["x"], dtype=float).reshape(-1)
        bias_state.linear_bias -= config.penalty_step * x
        _add_pair_penalty(bias_state.pair_bias, x, config.pair_penalty_step)
        bias_state.feasibility_penalty_scale *= config.infeasible_penalty_growth
        _push_tabu(bias_state, x, config)

    # 3) 新增：可行但差的候选，也进入 no-good 逻辑
    best_obj = float(best_record["objective_value"])
    bad_threshold = config.feasible_bad_ratio * best_obj
    incumbent = bias_state.incumbent.copy() if bias_state.incumbent is not None else None

    for rec in feasible_records:
        obj = float(rec["objective_value"])
        x = np.asarray(rec["x"], dtype=float).reshape(-1)

        if obj < bad_threshold:
            bias_state.linear_bias -= config.penalty_step * x
            _add_pair_penalty(bias_state.pair_bias, x, config.pair_penalty_step)

            # 若还离 incumbent 很近，说明在同一 basin 打转，更应推开
            if incumbent is not None:
                dist = int(np.sum(np.abs(x - incumbent) > 0.5))
                if dist <= max(1, stats.n // 6):
                    bias_state.linear_bias -= 0.5 * config.penalty_step * x
                    _add_pair_penalty(bias_state.pair_bias, x, 0.5 * config.pair_penalty_step)

            _push_tabu(bias_state, x, config)

    # 4) elite 高频结构轻量强化
    elite_candidates = feasible_records[: max(1, min(3, len(feasible_records)))]
    _update_elite_pool(bias_state, elite_candidates, config)

    if bias_state.elite_pool:
        freq = np.zeros(stats.n, dtype=float)
        for rec in bias_state.elite_pool:
            freq += np.asarray(rec["x"], dtype=float).reshape(-1)
        freq /= max(1, len(bias_state.elite_pool))
        bias_state.linear_bias += config.elite_reward_step * (freq - 0.5)

    _clip_bias(bias_state, config)
    return bias_state