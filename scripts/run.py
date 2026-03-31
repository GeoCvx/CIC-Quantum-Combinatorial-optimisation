# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

import numpy as np

from algorithms.pipeline.milp_pipeline import solve_milp
from algorithms.pipeline.micp_pipeline import solve_micp
from algorithms.subproblem.router import detect_problem_type

from algorithms.master.hybrid_master import HybridMasterConfig
from algorithms.master.quantum_master import QuantumMasterConfig
from algorithms.quantum.qubo_builder import QUBOConfig
from algorithms.quantum.qaoa_solver import QAOASolverConfig


# =========================
# 配置构造（可根据需要调）
# =========================

def make_quantum_config() -> QuantumMasterConfig:
    return QuantumMasterConfig(
        qubo_config=QUBOConfig(
            objective_scale=1.0,
            resource_penalty=10.0,
            hamming_penalty=0.0,
            demand_weight=1.0,
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


def make_hybrid_config() -> HybridMasterConfig:
    return HybridMasterConfig(
        use_classical=True,
        use_quantum=True,
        classical_num_starts=10,
        classical_local_iter=50,
        classical_seed=42,
        quantum_config=make_quantum_config(),
        fallback_to_classical=True,
    )


# =========================
# 主执行逻辑
# =========================

def run(
    input_path: pathlib.Path,
    master_mode: str = "hybrid",
    output_path: pathlib.Path | None = None,
):
    # ===== 1. 读取数据 =====
    with open(input_path, "r", encoding="utf-8") as f:
        problem_dict = json.load(f)

    # ===== 2. 自动识别问题类型 =====
    problem_type = detect_problem_type(problem_dict)

    print(f"\n=== Problem Info ===")
    print(f"file: {input_path}")
    print(f"type: {problem_type}")
    print(f"master_mode: {master_mode}")

    # ===== 3. 构造 master_config =====
    if master_mode == "quantum":
        master_config: Any = make_quantum_config()
    elif master_mode == "hybrid":
        master_config = make_hybrid_config()
    elif master_mode == "classical":
        master_config = {
            "num_starts": 10,
            "local_iter": 100,
            "seed": 42,
        }
    else:
        raise ValueError(f"不支持 master_mode: {master_mode}")

    # ===== 4. 调用 pipeline =====
    start = time.time()

    if problem_type == "MILP":
        result = solve_milp(
            problem_dict,
            master_mode=master_mode,
            master_config=master_config,
        )
    else:
        result = solve_micp(
            problem_dict,
            master_mode=master_mode,
            master_config=master_config,
        )

    end = time.time()

    # ===== 5. 输出结果 =====
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

    # ===== 6. 保存结果（可选）=====
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n结果已保存到: {output_path}")

    return result


# =========================
# CLI 入口
# =========================

def main():
    parser = argparse.ArgumentParser(description="CIC Solver Runner")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 JSON 文件路径",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["classical", "quantum", "hybrid"],
        help="选择 master 模式",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出结果 JSON 文件路径（可选）",
    )

    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output) if args.output else None

    run(
        input_path=input_path,
        master_mode=args.mode,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()