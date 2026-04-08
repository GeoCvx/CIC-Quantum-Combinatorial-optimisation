# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from algorithms.candidate.candidate_selector import CandidateSelectionConfig
from algorithms.feedback.bias_updater import BiasUpdateConfig
from algorithms.local_search.refiner import LocalRefineConfig
from algorithms.master.hybrid_master import HybridMasterConfig
from algorithms.master.iterative_hybrid_master import IterativeHybridMasterConfig
from algorithms.master.quantum_master import QuantumMasterConfig
from algorithms.quantum.qaoa_solver import QAOASolverConfig
from algorithms.quantum.qubo_builder import QUBOConfig


def _load_json(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError("配置文件顶层必须是 JSON object。")
    return data


def _subset(src: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {k: src[k] for k in keys if k in src}


def _to_1d_float_array(values: Any, field_name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{field_name} 不能为空。")
    return arr


def build_quantum_config(config_dict: dict[str, Any]) -> QuantumMasterConfig:
    quantum_dict = dict(config_dict.get("quantum_config", {}))
    qubo_dict = dict(quantum_dict.get("qubo_config", {}))
    qaoa_dict = dict(quantum_dict.get("qaoa_config", {}))

    qubo_config = QUBOConfig(
        **_subset(
            qubo_dict,
            [
                "objective_scale",
                "resource_penalty",
                "hamming_penalty",
                "demand_weight",
                "conflict_weight",
                "tabu_penalty",
                "elite_weight",
                "exploration_boost",
                "exploration_seed",
            ],
        )
    )

    qaoa_kwargs = _subset(
        qaoa_dict,
        [
            "p",
            "shots",
            "seed",
            "gamma_min",
            "gamma_max",
            "gamma_points",
            "beta_min",
            "beta_max",
            "beta_points",
            "per_run_top_k",
            "final_top_k",
        ],
    )

    if "gamma_values" in qaoa_dict:
        qaoa_kwargs["gamma_values"] = _to_1d_float_array(
            qaoa_dict["gamma_values"], "gamma_values"
        )
    if "beta_values" in qaoa_dict:
        qaoa_kwargs["beta_values"] = _to_1d_float_array(
            qaoa_dict["beta_values"], "beta_values"
        )

    qaoa_config = QAOASolverConfig(**qaoa_kwargs)

    return QuantumMasterConfig(
        qubo_config=qubo_config,
        qaoa_config=qaoa_config,
        **_subset(
            quantum_dict,
            [
                "round_digits",
                "candidate_top_k",
                "include_solver_best",
                "deduplicate_candidates",
                "add_mutations_from_best",
                "add_mutations_from_incumbent",
                "max_best_mutations",
                "max_incumbent_mutations",
            ],
        ),
    )


def build_iterative_hybrid_master_config(
    config_dict: dict[str, Any]
) -> HybridMasterConfig:
    mode = str(config_dict.get("mode", "iterative_hybrid")).lower()
    if mode != "iterative_hybrid":
        raise ValueError(
            f"当前 loader 只服务 iterative_hybrid，收到 mode={mode!r}"
        )

    iter_dict = dict(config_dict.get("iterative_config", {}))
    candidate_dict = dict(iter_dict.get("candidate_config", {}))
    bias_dict = dict(iter_dict.get("bias_update_config", {}))
    local_dict = dict(iter_dict.get("local_refine_config", {}))

    quantum_config = build_quantum_config(config_dict)

    iterative_config = IterativeHybridMasterConfig(
        max_rounds=int(iter_dict.get("max_rounds", 5)),
        no_improve_patience=int(iter_dict.get("no_improve_patience", 2)),
        use_classical_warm_start=bool(
            iter_dict.get("use_classical_warm_start", False)
        ),
        classical_num_starts=int(iter_dict.get("classical_num_starts", 5)),
        classical_local_iter=int(iter_dict.get("classical_local_iter", 20)),
        classical_seed=int(iter_dict.get("classical_seed", 42)),
        quantum_config=quantum_config,
        candidate_config=CandidateSelectionConfig(
            **_subset(
                candidate_dict,
                [
                    "top_k",
                    "min_hamming_distance",
                    "prefer_higher_probability",
                    "min_far_hamming_distance",
                    "min_far_candidates",
                    "tabu_hamming_threshold",
                    "objective_bonus",
                    "far_bonus",
                ],
            )
        ),
        bias_update_config=BiasUpdateConfig(
            **_subset(
                bias_dict,
                [
                    "reward_step",
                    "penalty_step",
                    "pair_penalty_step",
                    "elite_reward_step",
                    "feasible_bad_ratio",
                    "max_abs_linear_bias",
                    "max_abs_pair_bias",
                    "infeasible_penalty_growth",
                    "no_improve_regularization_decay",
                    "no_improve_feasibility_growth",
                    "max_tabu_size",
                    "elite_max_size",
                ],
            )
        ),
        local_refine_config=LocalRefineConfig(
            **_subset(
                local_dict,
                [
                    "max_iter",
                    "enable_swap",
                    "max_start_points",
                    "use_guided_order",
                ],
            )
        ),
    )

    return HybridMasterConfig(
        use_classical=True,
        use_quantum=True,
        classical_num_starts=iterative_config.classical_num_starts,
        classical_local_iter=iterative_config.classical_local_iter,
        classical_seed=iterative_config.classical_seed,
        quantum_config=quantum_config,
        fallback_to_classical=True,
        iterative=True,
        iterative_config=iterative_config,
    )


def load_iterative_hybrid_master_config(
    config_path: str | Path,
) -> tuple[dict[str, Any], HybridMasterConfig]:
    config_dict = _load_json(config_path)
    master_config = build_iterative_hybrid_master_config(config_dict)
    return config_dict, master_config