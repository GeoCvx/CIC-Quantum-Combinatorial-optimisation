# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import json
import time
import argparse
import pathlib
from typing import Any

import numpy as np

# =========================================================
# 把项目根目录加入 PYTHONPATH
# 兼容两种放置方式：
# 1) 放在项目根目录下
# 2) 放在 scripts/ 目录下
# =========================================================
_THIS_FILE = pathlib.Path(__file__).resolve()

ROOT = None
for cand in [_THIS_FILE.parent, _THIS_FILE.parent.parent]:
    if (cand / "algorithms").exists():
        ROOT = cand
        break

if ROOT is None:
    raise RuntimeError("未找到项目根目录（包含 algorithms/ 的目录）")

sys.path.append(str(ROOT))

from algorithms.subproblem.router import detect_problem_type
from algorithms.pipeline.milp_pipeline import solve_milp
from algorithms.pipeline.micp_pipeline import solve_micp


# =========================================================
# 当前 solver 配置
# =========================================================
def make_quantum_config():
    from algorithms.master.quantum_master import QuantumMasterConfig
    from algorithms.quantum.qubo_builder import QUBOConfig
    from algorithms.quantum.qaoa_solver import QAOASolverConfig

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


def make_iterative_hybrid_config():
    from algorithms.master.hybrid_master import HybridMasterConfig
    from algorithms.master.iterative_hybrid_master import IterativeHybridMasterConfig
    from algorithms.candidate.candidate_selector import CandidateSelectionConfig
    from algorithms.feedback.bias_updater import BiasUpdateConfig
    from algorithms.local_search.refiner import LocalRefineConfig

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


# =========================================================
# 跑当前 solver
# =========================================================
def solve_with_current_solver(problem_dict: dict, mode: str = "iterative_hybrid") -> dict[str, Any]:
    problem_type = detect_problem_type(problem_dict)

    if mode == "iterative_hybrid":
        master_config = make_iterative_hybrid_config()
    else:
        raise ValueError(f"当前 run 脚本只支持 mode=iterative_hybrid，收到: {mode}")

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


# =========================================================
# 输出格式处理
# 参考 example_answer.json:
# {"x": [...], "y": [...], "Z": 13170.0, "r": [...]}
# =========================================================
def _normalize_x(x: list[Any]) -> list[int]:
    return [int(round(float(v))) for v in x]


def _normalize_float_list(arr: list[Any], digits: int = 6) -> list[float]:
    out = []
    for v in arr:
        fv = float(v)
        if abs(fv - round(fv)) < 10 ** (-digits):
            fv = float(round(fv))
        else:
            fv = round(fv, digits)
        out.append(fv)
    return out


def format_solution_as_answer_json(solution: dict[str, Any], digits: int = 6) -> dict[str, Any]:
    x = _normalize_x(solution["x"])
    y = _normalize_float_list(solution["y"], digits=digits)
    Z = round(float(solution["Z"]), digits)
    r = _normalize_float_list(solution["r"], digits=digits)

    return {
        "x": x,
        "y": y,
        "Z": Z,
        "r": r,
    }


# =========================================================
# 单文件运行
# =========================================================
def run_one(input_path: pathlib.Path, output_path: pathlib.Path, mode: str = "iterative_hybrid") -> dict[str, Any]:
    with open(input_path, "r", encoding="utf-8") as f:
        problem_dict = json.load(f)

    start = time.perf_counter()
    result = solve_with_current_solver(problem_dict, mode=mode)
    elapsed = time.perf_counter() - start

    solution = result["solution"]
    answer_json = format_solution_as_answer_json(solution, digits=6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(answer_json, f, ensure_ascii=False)

    print(f"[OK] {input_path.name} -> {output_path.name} | "
          f"type={detect_problem_type(problem_dict)} | "
          f"Z={answer_json['Z']} | "
          f"runtime={round(float(result.get('runtime', elapsed)), 6)}s")

    return answer_json


# =========================================================
# 批量运行
# =========================================================
def run_batch(input_dir: pathlib.Path, output_dir: pathlib.Path, mode: str = "iterative_hybrid") -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是目录: {input_dir}")

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"目录下未找到任何 json 文件: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Batch Run Start ===")
    print("input_dir :", input_dir)
    print("output_dir:", output_dir)
    print("file_count:", len(json_files))
    print()

    succeeded = 0
    failed = 0

    for input_path in json_files:
        output_name = input_path.stem + "_answer.json"
        output_path = output_dir / output_name
        try:
            run_one(input_path=input_path, output_path=output_path, mode=mode)
            succeeded += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {input_path.name}: {repr(e)}")

    print()
    print("=== Batch Run Summary ===")
    print("succeeded:", succeeded)
    print("failed   :", failed)
    print("output_dir:", output_dir)


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run current solver and save result JSON(s) in example_answer.json format."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input",
        type=str,
        help="单个输入问题 JSON 路径，例如 data/raw/problem_micp_1.json",
    )
    group.add_argument(
        "--input_dir",
        type=str,
        help="批量输入目录，例如 data/raw",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="单文件输出 JSON 路径；仅和 --input 一起使用",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="批量输出目录；和 --input_dir 一起使用时默认 output/",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="iterative_hybrid",
        choices=["iterative_hybrid"],
        help="当前仅支持 iterative_hybrid",
    )

    args = parser.parse_args()

    if args.input is not None:
        input_path = pathlib.Path(args.input)
        if args.output is None:
            output_dir = ROOT / "output"
            output_name = input_path.stem + "_answer.json"
            output_path = output_dir / output_name
        else:
            output_path = pathlib.Path(args.output)

        run_one(input_path=input_path, output_path=output_path, mode=args.mode)
        return

    input_dir = pathlib.Path(args.input_dir)
    if args.output_dir is None:
        output_dir = ROOT / "output"
    else:
        output_dir = pathlib.Path(args.output_dir)

    run_batch(input_dir=input_dir, output_dir=output_dir, mode=args.mode)


if __name__ == "__main__":
    main()
