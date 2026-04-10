# -*- coding: utf-8 -*-
from __future__ import annotations

import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.run_with_config import run as run_with_config


# =========================
# 路径配置
# =========================
DATA_DIR = ROOT / "data" / "raw"
CONFIG_ROOT = ROOT / "config"
OUTPUT_ROOT = ROOT / "output" / "param_comp"


# =========================
# 核心函数
# =========================
def run_batch_for_module(module_name: str):
    config_dir = CONFIG_ROOT / module_name
    output_dir = OUTPUT_ROOT / module_name

    if not config_dir.exists():
        raise FileNotFoundError(f"config 目录不存在: {config_dir}")

    problem_files = sorted(DATA_DIR.glob("*.json"))
    config_files = sorted(config_dir.glob("*.json"))

    print("\n==============================")
    print(f"Module: {module_name}")
    print(f"Problems: {len(problem_files)}")
    print(f"Configs : {len(config_files)}")
    print("==============================\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_jobs = len(problem_files) * len(config_files)
    job_id = 0

    for cfg_path in config_files:
        cfg_name = cfg_path.stem

        for prob_path in problem_files:
            job_id += 1

            prob_name = prob_path.stem
            out_name = f"{cfg_name}__{prob_name}.json"
            out_path = output_dir / out_name

            print(f"[{job_id}/{total_jobs}] Running:")
            print(f"  config : {cfg_name}")
            print(f"  problem: {prob_name}")

            t0 = time.time()

            try:
                run_with_config(
                    input_path=prob_path,
                    config_path=cfg_path,
                    output_path=out_path,
                )

                dt = time.time() - t0
                print(f"  ✔ Done in {dt:.3f}s\n")

            except Exception as e:
                print(f"  ✘ Failed: {repr(e)}\n")


# =========================
# CLI
# =========================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch run ablation experiments for a given module"
    )
    parser.add_argument(
        "--module",
        type=str,
        required=True,
        help="模块名，例如 qubo / qaoa / candidate / bias / local",
    )

    args = parser.parse_args()

    run_batch_for_module(args.module)


if __name__ == "__main__":
    main()