# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import json
import argparse
import pathlib
from typing import Any

import numpy as np

# ===== 把项目根目录加入 PYTHONPATH =====
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from algorithms.subproblem.router import detect_problem_type, evaluate_subproblem
from algorithms.pipeline.milp_pipeline import solve_milp
from algorithms.pipeline.micp_pipeline import solve_micp
from algorithms.master.hybrid_master import HybridMasterConfig
from algorithms.master.iterative_hybrid_master import IterativeHybridMasterConfig
from algorithms.master.quantum_master import QuantumMasterConfig
from algorithms.quantum.qubo_builder import QUBOConfig
from algorithms.quantum.qaoa_solver import QAOASolverConfig
from algorithms.candidate.candidate_selector import CandidateSelectionConfig
from algorithms.feedback.bias_updater import BiasUpdateConfig
from algorithms.local_search.refiner import LocalRefineConfig


# =========================
# 官方 baseline（改成可直接复用）
# =========================
def solve_with_greedy(example: dict) -> dict:
    n = example["product_count"]
    m = example["resource_count"]
    p = np.array(example["price"], dtype=float)
    f = np.array(example["fixed_cost"], dtype=float)
    alpha = np.array(example["alpha"], dtype=float)
    beta = np.array(example["beta"], dtype=float)
    D = np.array(example["max_demand"], dtype=float)
    R = np.array(example["resource_limit"], dtype=float)
    A = np.array(example["consumption_matrix"], dtype=float)

    # 1) 单产品利润顶点
    y_peak = np.zeros(n, dtype=float)
    for i in range(n):
        if beta[i] > 0:
            y_peak[i] = max(0.0, (p[i] - alpha[i]) / (2.0 * beta[i]))
        else:
            y_peak[i] = float("inf") if (p[i] - alpha[i]) > 0 else 0.0

    # 2) 理想利润估计
    ideal_profits = np.zeros(n, dtype=float)
    for i in range(n):
        max_y_physical = min(
            [R[j] / A[j, i] if A[j, i] > 0 else float("inf") for j in range(m)]
        )
        best_y_ideal = min(D[i], y_peak[i], max_y_physical)
        prof = p[i] * best_y_ideal - (
            alpha[i] * best_y_ideal + beta[i] * (best_y_ideal ** 2)
        ) - f[i]
        ideal_profits[i] = prof

    # 3) 按理想利润排序
    sorted_indices = np.argsort(ideal_profits)[::-1]

    # 4) 贪心分配
    current_R = R.copy()
    x_greedy = np.zeros(n, dtype=int)
    y_greedy = np.zeros(n, dtype=float)
    total_profit = 0.0

    for i in sorted_indices:
        if ideal_profits[i] <= 0:
            continue

        max_y_actual = min(
            [current_R[j] / A[j, i] if A[j, i] > 0 else float("inf") for j in range(m)]
        )
        best_y_actual = min(D[i], y_peak[i], max_y_actual)

        actual_prof = p[i] * best_y_actual - (
            alpha[i] * best_y_actual + beta[i] * (best_y_actual ** 2)
        ) - f[i]

        if actual_prof > 0:
            x_greedy[i] = 1
            y_greedy[i] = best_y_actual
            total_profit += actual_prof
            for j in range(m):
                current_R[j] -= A[j, i] * best_y_actual

    return {
        "x": x_greedy.astype(float).tolist(),
        "y": np.round(y_greedy, 6).tolist(),
        "Z": float(total_profit),
        "r": np.round(current_R, 6).tolist(),
    }


# =========================
# 你的当前 solver 配置
# =========================
def make_quantum_config() -> QuantumMasterConfig:
    return QuantumMasterConfig(
        qubo_config=QUBOConfig(
            objective_scale=1.0,
            resource_penalty=10.0,
            hamming_penalty=0.5,
            conflict_weight=1.0,
            tabu_penalty=5.0,
            elite_weight=0.5,
            exploration_boost=0.5,
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
        candidate_top_k=8,
        include_solver_best=True,
        deduplicate_candidates=True,
        add_mutations_from_best=True,
        add_mutations_from_incumbent=True,
        max_best_mutations=3,
        max_incumbent_mutations=3,
    )


def make_iterative_hybrid_config() -> HybridMasterConfig:
    iterative_cfg = IterativeHybridMasterConfig(
        max_rounds=3,
        no_improve_patience=2,
        classical_num_starts=5,
        classical_local_iter=20,
        classical_seed=42,
        quantum_config=make_quantum_config(),
        candidate_config=CandidateSelectionConfig(
            top_k=8,
            min_hamming_distance=2,
            prefer_higher_probability=True,
            min_far_hamming_distance=3,
            min_far_candidates=2,
            tabu_hamming_threshold=1,
            objective_bonus=0.05,
            far_bonus=0.02,
        ),
        bias_update_config=BiasUpdateConfig(),
        local_refine_config=LocalRefineConfig(
            max_iter=20,
            enable_swap=True,
            max_start_points=3,
            use_guided_order=True,
        ),
    )

    return HybridMasterConfig(
        use_classical=True,
        use_quantum=True,
        classical_num_starts=5,
        classical_local_iter=20,
        classical_seed=42,
        quantum_config=make_quantum_config(),
        fallback_to_classical=True,
        iterative=True,
        iterative_config=iterative_cfg,
    )


# =========================
# 跑你当前 solver
# =========================
def solve_with_current_solver(problem_dict: dict, mode: str = "iterative_hybrid") -> dict[str, Any]:
    problem_type = detect_problem_type(problem_dict)

    if mode == "iterative_hybrid":
        master_config = make_iterative_hybrid_config()
    else:
        raise ValueError(f"当前 compare 脚本只支持 mode=iterative_hybrid，收到: {mode}")

    if problem_type == "MILP":
        result = solve_milp(
            problem_dict,
            master_mode=mode,
            master_config=master_config,
        )
    else:
        result = solve_micp(
            problem_dict,
            master_mode=mode,
            master_config=master_config,
        )

    return result


# =========================
# 用统一评估器复核 baseline
# =========================
def evaluate_baseline_fairly(problem_dict: dict, baseline_result: dict) -> dict[str, Any]:
    x = np.asarray(baseline_result["x"], dtype=float)
    fair_eval = evaluate_subproblem(problem_dict, x, round_digits=6)
    return {
        "x": fair_eval["x"],
        "y": fair_eval["y"],
        "Z": fair_eval["Z"],
        "r": fair_eval["r"],
    }


# =========================
# 打印
# =========================
def print_solution_block(title: str, sol: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print("x:", sol["x"])
    print("y:", sol["y"])
    print("Z:", sol["Z"])
    print("r:", sol["r"])


def compare(problem_path: pathlib.Path) -> None:
    with open(problem_path, "r", encoding="utf-8") as f:
        problem_dict = json.load(f)

    print("=== Problem Info ===")
    print("file:", problem_path)
    print("type:", detect_problem_type(problem_dict))

    # baseline
    baseline_raw = solve_with_greedy(problem_dict)
    baseline_fair = evaluate_baseline_fairly(problem_dict, baseline_raw)

    # current solver
    current_result = solve_with_current_solver(problem_dict, mode="iterative_hybrid")
    current_sol = current_result["solution"]

    print_solution_block("Baseline (raw greedy)", baseline_raw)
    print_solution_block("Baseline (fair eval by project evaluator)", baseline_fair)
    print_solution_block("Current Solver", current_sol)

    z_base_raw = float(baseline_raw["Z"])
    z_base_fair = float(baseline_fair["Z"])
    z_current = float(current_sol["Z"])

    print("\n=== Comparison Summary ===")
    print(f"Baseline raw Z        : {z_base_raw:.6f}")
    print(f"Baseline fair-eval Z  : {z_base_fair:.6f}")
    print(f"Current solver Z      : {z_current:.6f}")
    print(f"Current - Baseline(raw): {z_current - z_base_raw:.6f}")
    print(f"Current - Baseline(fair): {z_current - z_base_fair:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Compare official baseline and current solver")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="问题 JSON 路径，例如 data/raw/problem_milp_2.json",
    )
    args = parser.parse_args()

    compare(pathlib.Path(args.input))


if __name__ == "__main__":
    main()