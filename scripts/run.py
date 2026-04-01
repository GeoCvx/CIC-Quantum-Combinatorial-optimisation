# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
import json
import time
from typing import Any
import numpy as np

from algorithms.pipeline.milp_pipeline import solve_milp
from algorithms.pipeline.micp_pipeline import solve_micp
from algorithms.subproblem.router import detect_problem_type
from algorithms.master.hybrid_master import HybridMasterConfig
from algorithms.master.iterative_hybrid_master import IterativeHybridMasterConfig
from algorithms.master.quantum_master import QuantumMasterConfig
from algorithms.quantum.qubo_builder import QUBOConfig
from algorithms.quantum.qaoa_solver import QAOASolverConfig


def make_quantum_config() -> QuantumMasterConfig:
    return QuantumMasterConfig(
        qubo_config=QUBOConfig(
            objective_scale=1.0,
            resource_penalty=10.0,
            hamming_penalty=1.5,
            conflict_weight=1.0,
            tabu_penalty=5.0,
        ),
        qaoa_config=QAOASolverConfig(
            p=1,
            shots=128,
            gamma_values=np.array([0.2, 0.5], dtype=float),
            beta_values=np.array([0.2, 0.5], dtype=float),
            seed=42,
            per_run_top_k=5,
            final_top_k=10,
        ),
        round_digits=6,
        candidate_top_k=5,
        include_solver_best=True,
        deduplicate_candidates=True,
    )


def make_hybrid_config(iterative: bool = False) -> HybridMasterConfig:
    quantum_cfg = make_quantum_config()
    cfg = HybridMasterConfig(
        use_classical=True,
        use_quantum=True,
        classical_num_starts=5,
        classical_local_iter=30,
        classical_seed=42,
        quantum_config=quantum_cfg,
        fallback_to_classical=True,
        iterative=iterative,
    )
    if iterative:
        cfg.iterative_config = IterativeHybridMasterConfig(
            max_rounds=4,
            no_improve_patience=2,
            classical_num_starts=5,
            classical_local_iter=20,
            classical_seed=42,
            quantum_config=quantum_cfg,
        )
    return cfg


def run(input_path: pathlib.Path, master_mode: str = "hybrid", output_path: pathlib.Path | None = None):
    with open(input_path, "r", encoding="utf-8") as f:
        problem_dict = json.load(f)

    problem_type = detect_problem_type(problem_dict)
    print(f"\n=== Problem Info ===")
    print(f"file: {input_path}")
    print(f"type: {problem_type}")
    print(f"master_mode: {master_mode}")

    if master_mode == "quantum":
        master_config: Any = make_quantum_config()
    elif master_mode == "hybrid":
        master_config = make_hybrid_config(iterative=False)
    elif master_mode == "iterative_hybrid":
        master_config = make_hybrid_config(iterative=True)
    elif master_mode == "classical":
        master_config = {"num_starts": 5, "local_iter": 30, "seed": 42}
    else:
        raise ValueError(f"不支持 master_mode: {master_mode}")

    start = time.time()
    if problem_type == "MILP":
        result = solve_milp(problem_dict, master_mode=master_mode, master_config=master_config)
    else:
        result = solve_micp(problem_dict, master_mode=master_mode, master_config=master_config)
    end = time.time()

    print("\n=== Result ===")
    print("status:", result["status"])
    print("objective_value:", result["objective_value"])
    print("runtime (pipeline):", result["runtime"])
    print("runtime (total):", end - start)
    sol = result["solution"]
    print("\n--- Solution ---")
    print("x:", sol["x"])
    print("y:", sol["y"])
    print("Z:", sol["Z"])
    print("r:", sol["r"])
    print("\n--- Extra ---")
    print(result["extra"])

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n结果已保存到: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="CIC Solver Runner")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--mode", type=str, default="hybrid", choices=["classical", "quantum", "hybrid", "iterative_hybrid"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run(pathlib.Path(args.input), master_mode=args.mode, output_path=pathlib.Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
