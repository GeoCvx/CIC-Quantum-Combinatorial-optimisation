# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from algorithms.preprocess.feature_builder import build_problem_stats
from algorithms.feedback.bias_updater import (
    BiasUpdateConfig,
    init_bias_state,
    update_bias_state,
)
from algorithms.candidate.candidate_selector import (
    CandidateSelectionConfig,
    select_candidates,
)
from algorithms.local_search.refiner import LocalRefineConfig, refine_solution_locally
from algorithms.master.classical_master import search_x_classical
from algorithms.master.quantum_master import QuantumMasterConfig, search_x_quantum
from algorithms.quantum.qubo_builder import QUBOConfig
from algorithms.subproblem.router import evaluate_subproblem


@dataclass
class IterativeHybridMasterConfig:
    max_rounds: int = 5
    no_improve_patience: int = 2

    # 新增：是否使用 classical warm start
    use_classical_warm_start: bool = False

    classical_num_starts: int = 5
    classical_local_iter: int = 20
    classical_seed: int = 42

    quantum_config: QuantumMasterConfig | None = None
    candidate_config: CandidateSelectionConfig = field(default_factory=CandidateSelectionConfig)
    bias_update_config: BiasUpdateConfig = field(default_factory=BiasUpdateConfig)
    local_refine_config: LocalRefineConfig = field(default_factory=LocalRefineConfig)

def _make_zero_init(problem_dict: dict, round_digits: int = 6) -> dict[str, Any]:
    n = int(problem_dict["product_count"])
    x0 = np.zeros(n, dtype=float)
    res0 = evaluate_subproblem(problem_dict, x0, round_digits=round_digits)
    return {
        "x": np.asarray(res0["x"], dtype=float).copy(),
        "best_result": res0,
        "objective_value": float(res0["Z"]),
        "feasible": True,
        "source": "all_zero_init",
    }


def search_x_hybrid_iterative(
    problem_dict: dict,
    config: IterativeHybridMasterConfig | None = None,
) -> dict[str, Any]:
    if config is None:
        config = IterativeHybridMasterConfig()

    stats = build_problem_stats(problem_dict)
    bias_state = init_bias_state(stats)

    classical_result = None  # 关键：先给默认值

    round_digits = 6
    if config.quantum_config is not None:
        round_digits = int(config.quantum_config.round_digits)

    # ========= 初始化 current_best =========
    # 方案 A：保留 classical warm start
    if config.use_classical_warm_start:
        classical_result = search_x_classical(
            problem_dict,
            num_starts=config.classical_num_starts,
            local_iter=config.classical_local_iter,
            seed=config.classical_seed,
        )
        current_best = {
            "x": np.asarray(classical_result["best_x"], dtype=float).copy(),
            "best_result": classical_result["best_result"],
            "objective_value": float(classical_result["best_objective"]),
            "feasible": True,
            "source": "classical_warm_start",
        }
        bias_state.incumbent = current_best["x"].copy()
    # 方案 B：去掉 warm start，直接从 all-zero 基线开始
    else:
        current_best = _make_zero_init(problem_dict, round_digits=round_digits)
        # 关键：不把全零解当成 incumbent
        # 第一轮 quantum 直接在 incumbent=None 的情况下自由搜索
        bias_state.incumbent = None

    rounds: list[dict[str, Any]] = []

    for t in range(config.max_rounds):
        previous_best_objective = float(current_best["objective_value"])
        previous_incumbent = bias_state.incumbent.copy() if bias_state.incumbent is not None else None

        quantum_config = config.quantum_config or QuantumMasterConfig()
        if quantum_config.qubo_config is None:
            quantum_config.qubo_config = QUBOConfig()

        # ===== quantum search =====
        quantum_result = search_x_quantum(
            problem_dict,
            config=quantum_config,
            incumbent=bias_state.incumbent,
            stats=stats,
            bias_state=bias_state,
        )

        raw_candidates = [
            {
                "bitstring": rec["bitstring"],
                "x": np.asarray(rec["x"], dtype=float),
                "energy": rec.get("energy"),
                "probability": rec.get("probability"),
                "source": rec.get("source", "quantum"),
                "objective_value": float(rec["objective_value"]),
                "feasible": bool(rec.get("feasible", True)),
                "subproblem_result": rec.get("subproblem_result"),
            }
            for rec in quantum_result["evaluated_candidates"]
        ]

        selected = select_candidates(
            raw_candidates,
            config=config.candidate_config,
            incumbent=bias_state.incumbent,
            tabu_set=bias_state.tabu_set,
        )
        selected_keys = {item.get("bitstring") for item in selected}

        selected_evaluated = []
        for rec in raw_candidates:
            if rec["bitstring"] not in selected_keys:
                continue
            selected_evaluated.append(
                {
                    "bitstring": rec["bitstring"],
                    "x": np.asarray(rec["x"], dtype=float).copy(),
                    "objective_value": float(rec["objective_value"]),
                    "feasible": bool(rec.get("feasible", True)),
                    "source": rec.get("source", "quantum"),
                    "subproblem_result": rec.get("subproblem_result"),
                }
            )

        # ===== 先用 selected 尝试更新 best =====
        best_eval = None
        if selected_evaluated:
            best_eval = max(selected_evaluated, key=lambda item: item["objective_value"])
            if best_eval["objective_value"] > previous_best_objective:
                current_best = {
                    "x": best_eval["x"].copy(),
                    "best_result": best_eval.get("subproblem_result"),
                    "objective_value": best_eval["objective_value"],
                    "feasible": True,
                    "source": "quantum_selected",
                }
                bias_state.incumbent = best_eval["x"].copy()

        # ===== 用全部已评估候选做反馈，而不是只用 selected =====
        bias_state = update_bias_state(
            bias_state=bias_state,
            evaluated_candidates=raw_candidates,
            previous_best_objective=previous_best_objective,
            stats=stats,
            config=config.bias_update_config,
        )

        # ===== local refine =====
        refine_start = bias_state.incumbent if bias_state.incumbent is not None else current_best["x"]
        refined = refine_solution_locally(
            problem_dict,
            refine_start,
            config=config.local_refine_config,
            linear_bias=bias_state.linear_bias, #修改：（添加linear_bias=bias_state.linear_bias,）当前主循环已经先做了 update_bias_state(...)，说明这一轮已经学到了新的 linear_bias；但后面的 local refine 没接这个 bias，相当于反馈信息没传到最后一层精修。
        )

        if refined["objective_value"] > current_best["objective_value"]:
            current_best = {
                "x": refined["x"].copy(),
                "best_result": refined["best_result"],
                "objective_value": refined["objective_value"],
                "feasible": True,
                "source": "local_refine",
            }
            bias_state.incumbent = refined["x"].copy()

        incumbent_changed = False
        if previous_incumbent is None and bias_state.incumbent is not None:
            incumbent_changed = True
        elif previous_incumbent is not None and bias_state.incumbent is not None:
            incumbent_changed = not np.allclose(previous_incumbent, bias_state.incumbent)

        rounds.append(
            {
                "round": t,
                "best_objective": float(current_best["objective_value"]),
                "best_source": current_best["source"],
                "num_selected_candidates": len(selected),
                "num_evaluated_candidates": len(raw_candidates),
                "no_improve_rounds": bias_state.no_improve_rounds,
                "tabu_size": len(bias_state.tabu_set),
                "elite_size": len(bias_state.elite_pool),
                "incumbent_changed": incumbent_changed,
            }
        )

        if bias_state.no_improve_rounds >= config.no_improve_patience:
            break

    # ===== 若 current_best 没带完整结果，则补评估一次 =====
    if current_best.get("best_result") is None:
        best_result = evaluate_subproblem(
            problem_dict,
            current_best["x"],
            round_digits=round_digits,
        )
    else:
        best_result = current_best["best_result"]

    return {
        "best_x": np.asarray(current_best["x"], dtype=float).copy(),
        "best_result": best_result,
        "best_objective": float(best_result["Z"]),
        "selected_master": "iterative_hybrid",
        "classical_result": classical_result,
        "quantum_result": None,
        "extra": {
            "rounds": rounds,
            "elite_size": len(bias_state.elite_pool),
            "tabu_size": len(bias_state.tabu_set),
            "has_incumbent": bias_state.incumbent is not None,
            "best_source": current_best["source"],
            "iterative": True,
        },
    }