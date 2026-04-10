# -*- coding: utf-8 -*-
'''
作用：

从 JSON 读配置
跑 iterative_hybrid
输出文件名自动和 config 文件绑定
同时把 config snapshot 写进输出结果，方便后续做稳定性分析
'''
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

from algorithms.pipeline.micp_pipeline import solve_micp
from algorithms.pipeline.milp_pipeline import solve_milp
from algorithms.subproblem.router import detect_problem_type
from config.default import OUTPUT_DIR
from config.json_loader import load_iterative_hybrid_master_config


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pathlib.Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _make_output_path(
    input_path: pathlib.Path,
    config_path: pathlib.Path,
    mode: str,
) -> pathlib.Path:
    out_dir = ROOT / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{input_path.stem}__{mode}__cfg-{config_path.stem}.json"
    return out_dir / filename


def run(
    input_path: pathlib.Path,
    config_path: pathlib.Path,
    output_path: pathlib.Path | None = None,
):
    with input_path.open("r", encoding="utf-8") as f:
        problem_dict = json.load(f)

    config_dict, master_config = load_iterative_hybrid_master_config(config_path)
    mode = str(config_dict.get("mode", "iterative_hybrid")).lower()
    problem_type = detect_problem_type(problem_dict)

    print("\n=== Problem Info ===")
    print(f"file: {input_path}")
    print(f"type: {problem_type}")
    print(f"mode: {mode}")
    print(f"config: {config_path}")

    start = time.time()
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
    end = time.time()

    result.setdefault("extra", {})
    result["extra"]["config_file"] = str(config_path)
    result["extra"]["config_name"] = config_path.name
    result["extra"]["config_snapshot"] = config_dict
    result["extra"]["total_runtime"] = end - start

    print("\n=== Result ===")
    print("status:", result["status"])
    print("objective_value:", result["objective_value"])
    print("runtime (pipeline):", result["runtime"])
    print("runtime (total):", end - start)

    if output_path is None:
        output_path = _make_output_path(
            input_path=input_path,
            config_path=config_path,
            mode=mode,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )

    print(f"\n结果已保存到: {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run CIC solver from JSON config."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run(
        input_path=pathlib.Path(args.input),
        config_path=pathlib.Path(args.config),
        output_path=pathlib.Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()