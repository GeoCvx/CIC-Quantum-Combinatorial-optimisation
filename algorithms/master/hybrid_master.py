# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

from algorithms.master.classical_master import search_x_classical
from algorithms.master.quantum_master import QuantumMasterConfig, search_x_quantum
from algorithms.master.iterative_hybrid_master import (
    IterativeHybridMasterConfig,
    search_x_hybrid_iterative,
)
from algorithms.preprocess.feature_builder import build_problem_stats
from algorithms.feedback.bias_updater import init_bias_state


@dataclass
class HybridMasterConfig:
    use_classical: bool = True
    use_quantum: bool = True
    classical_num_starts: int = 10
    classical_local_iter: int = 100
    classical_seed: int = 42
    quantum_config: QuantumMasterConfig | None = None
    fallback_to_classical: bool = True
    iterative: bool = False
    iterative_config: IterativeHybridMasterConfig | None = None


def _safe_objective(result: Dict[str, Any] | None) -> float:
    if result is None:
        return -float("inf")
    return float(result.get("best_objective", -float("inf")))


def search_x_hybrid(problem_dict: dict, config: HybridMasterConfig | None = None) -> Dict[str, Any]:
    if config is None:
        config = HybridMasterConfig()

    if config.iterative:
        iterative_config = config.iterative_config or IterativeHybridMasterConfig(
            classical_num_starts=config.classical_num_starts,
            classical_local_iter=config.classical_local_iter,
            classical_seed=config.classical_seed,
            quantum_config=config.quantum_config,
        )
        return search_x_hybrid_iterative(problem_dict, iterative_config)

    classical_result: Dict[str, Any] | None = None
    quantum_result: Dict[str, Any] | None = None
    quantum_error: str | None = None

    incumbent: np.ndarray | None = None
    stats = build_problem_stats(problem_dict)
    bias_state = init_bias_state(stats)

    if config.use_classical:
        classical_result = search_x_classical(
            problem_dict,
            num_starts=config.classical_num_starts,
            local_iter=config.classical_local_iter,
            seed=config.classical_seed,
        )
        if classical_result is not None and classical_result.get("best_x") is not None:
            incumbent = np.asarray(classical_result["best_x"], dtype=float).copy()
            bias_state.incumbent = incumbent.copy()

    if config.use_quantum:
        try:
            quantum_result = search_x_quantum(
                problem_dict,
                config=config.quantum_config,
                incumbent=incumbent,
                stats=stats,
                bias_state=bias_state,
            )
        except Exception as e:
            quantum_error = repr(e)
            if not config.fallback_to_classical:
                raise

    classical_obj = _safe_objective(classical_result)
    quantum_obj = _safe_objective(quantum_result)

    if quantum_obj > classical_obj:
        final_result = quantum_result
        selected_master = "quantum"
    else:
        final_result = classical_result
        selected_master = "classical"

    if final_result is None:
        raise RuntimeError("hybrid master 未能得到任何有效结果。")

    return {
        "best_x": final_result["best_x"],
        "best_result": final_result["best_result"],
        "best_objective": final_result["best_objective"],
        "selected_master": selected_master,
        "classical_result": classical_result,
        "quantum_result": quantum_result,
        "extra": {
            "classical_objective": classical_obj,
            "quantum_objective": quantum_obj,
            "used_classical": config.use_classical,
            "used_quantum": config.use_quantum,
            "quantum_error": quantum_error,
            "has_incumbent": incumbent is not None,
            "iterative": False,
        },
    }
