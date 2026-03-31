# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

from algorithms.master.classical_master import search_x_classical
from algorithms.master.quantum_master import (
    QuantumMasterConfig,
    search_x_quantum,
)


@dataclass
class HybridMasterConfig:
    """
    混合主问题配置。

    use_classical:
        是否先跑经典 master。

    use_quantum:
        是否跑量子 refinement。

    classical_num_starts:
        经典多起点次数。

    classical_local_iter:
        经典局部搜索迭代次数。

    classical_seed:
        经典随机种子。

    quantum_config:
        量子 master 配置。

    fallback_to_classical:
        若量子阶段失败，是否回退到经典结果。
    """
    use_classical: bool = True
    use_quantum: bool = True

    classical_num_starts: int = 10
    classical_local_iter: int = 100
    classical_seed: int = 42

    quantum_config: QuantumMasterConfig | None = None

    fallback_to_classical: bool = True


def _safe_objective(result: Dict[str, Any] | None) -> float:
    if result is None:
        return -float("inf")
    return float(result.get("best_objective", -float("inf")))


def search_x_hybrid(
    problem_dict: dict,
    config: HybridMasterConfig | None = None,
) -> Dict[str, Any]:
    """
    hybrid master 主入口。

    流程：
    1) classical master 找到 baseline incumbent
    2) quantum master 以 incumbent 为引导做 refinement
    3) 按真实目标值 best_objective 选择更优结果
    """
    if config is None:
        config = HybridMasterConfig()

    classical_result: Dict[str, Any] | None = None
    quantum_result: Dict[str, Any] | None = None
    quantum_error: str | None = None

    # ===== 1. classical baseline =====
    incumbent: np.ndarray | None = None
    if config.use_classical:
        classical_result = search_x_classical(
            problem_dict,
            num_starts=config.classical_num_starts,
            local_iter=config.classical_local_iter,
            seed=config.classical_seed,
        )
        if classical_result is not None and classical_result.get("best_x") is not None:
            incumbent = np.asarray(classical_result["best_x"], dtype=float).copy()

    # ===== 2. quantum refinement =====
    if config.use_quantum:
        try:
            quantum_result = search_x_quantum(
                problem_dict,
                config=config.quantum_config,
                incumbent=incumbent,
            )
        except Exception as e:
            quantum_error = repr(e)
            if not config.fallback_to_classical:
                raise

    # ===== 3. select final best =====
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
        },
    }


if __name__ == "__main__":
    import json
    from algorithms.quantum.qubo_builder import QUBOConfig
    from algorithms.quantum.qaoa_solver import QAOASolverConfig

    with open("../../data/raw/problem_micp_1.json", "r", encoding="utf-8") as f:
        problem_dict = json.load(f)

    hybrid_config = HybridMasterConfig(
        use_classical=True,
        use_quantum=True,
        classical_num_starts=10,
        classical_local_iter=50,
        classical_seed=42,
        quantum_config=QuantumMasterConfig(
            qubo_config=QUBOConfig(
                objective_scale=1.0,
                resource_penalty=10.0,
                hamming_penalty=2.0,
                demand_weight=1.0,
            ),
            qaoa_config=QAOASolverConfig(
                p=1,
                shots=256,
                gamma_values=np.array([0.2, 0.5, 0.8], dtype=float),
                beta_values=np.array([0.2, 0.5, 0.8], dtype=float),
                seed=42,
                per_run_top_k=5,
                final_top_k=10,
            ),
            round_digits=6,
            candidate_top_k=5,
            include_solver_best=True,
            deduplicate_candidates=True,
        ),
        fallback_to_classical=True,
    )

    result = search_x_hybrid(problem_dict, config=hybrid_config)

    print("=== Hybrid Master Best ===")
    print("selected_master:", result["selected_master"])
    print("best_objective:", result["best_objective"])
    print("best_x:", result["best_x"].tolist())
    print("best_result:", result["best_result"])

    print("\n=== Summary ===")
    print(result["extra"])