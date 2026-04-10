# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import copy
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

BASE_CONFIG_PATH = ROOT / "config" / "iterative_hybrid_base.json"
OUTPUT_ROOT = ROOT / "config"


# =========================
# 工具函数
# =========================
def load_base():
    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: dict, subdir: str, name: str):
    out_dir = OUTPUT_ROOT / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"[OK] {path}")


def clone(cfg):
    return copy.deepcopy(cfg)


# =========================
# 1️⃣ QUBO 模块
# =========================
def generate_qubo(cfg):
    base = cfg["quantum_config"]["qubo_config"]

    sweep = {
        "resource_penalty": [5.0, 10.0, 20.0],
        "hamming_penalty": [0.5, 1.5, 3.0],
        "tabu_penalty": [2.0, 5.0, 10.0],
        "elite_weight": [0.2, 0.5, 1.0],
        "exploration_boost": [0.2, 0.5, 1.0],
    }

    for param, values in sweep.items():
        for v in values:
            new_cfg = clone(cfg)
            new_cfg["quantum_config"]["qubo_config"][param] = v

            name = f"qubo_{param}_{v}"
            save_config(new_cfg, "qubo", name)


# =========================
# 2️⃣ QAOA 模块
# =========================
def generate_qaoa(cfg):
    sweep = {
        "shots": [64, 128, 256],
        "p": [1, 2],
        "gamma_values": [[0.1, 0.3], [0.2, 0.5], [0.5, 1.0]],
        "beta_values": [[0.1, 0.3], [0.2, 0.5], [0.5, 1.0]],
    }

    for param, values in sweep.items():
        for v in values:
            new_cfg = clone(cfg)
            new_cfg["quantum_config"]["qaoa_config"][param] = v

            name = f"qaoa_{param}_{str(v).replace(' ', '')}"
            save_config(new_cfg, "qaoa", name)


# =========================
# 3️⃣ Candidate 模块
# =========================
def generate_candidate(cfg):
    sweep = {
        "top_k": [4, 8, 12],
        "min_hamming_distance": [1, 2, 3],
        "min_far_candidates": [1, 2, 4],
        "objective_bonus": [0.01, 0.05, 0.1],
        "far_bonus": [0.0, 0.02, 0.05],
    }

    for param, values in sweep.items():
        for v in values:
            new_cfg = clone(cfg)
            new_cfg["iterative_config"]["candidate_config"][param] = v

            name = f"candidate_{param}_{v}"
            save_config(new_cfg, "candidate", name)


# =========================
# 4️⃣ Bias Update 模块
# =========================
def generate_bias(cfg):
    sweep = {
        "reward_step": [0.05, 0.1, 0.2],
        "penalty_step": [0.1, 0.2, 0.3],
        "pair_penalty_step": [0.1, 0.15, 0.3],
        "elite_reward_step": [0.02, 0.05, 0.1],
    }

    for param, values in sweep.items():
        for v in values:
            new_cfg = clone(cfg)
            new_cfg["iterative_config"]["bias_update_config"][param] = v

            name = f"bias_{param}_{v}"
            save_config(new_cfg, "bias", name)


# =========================
# 5️⃣ Local Search 模块
# =========================
def generate_local(cfg):
    sweep = {
        "max_iter": [10, 20, 50],
        "max_start_points": [1, 3, 5],
    }

    for param, values in sweep.items():
        for v in values:
            new_cfg = clone(cfg)
            new_cfg["iterative_config"]["local_refine_config"][param] = v

            name = f"local_{param}_{v}"
            save_config(new_cfg, "local", name)


# =========================
# 主函数
# =========================
def main():
    cfg = load_base()

    # 确保 seed 固定（非常关键）
    cfg["quantum_config"]["qaoa_config"]["seed"] = 42
    cfg["quantum_config"]["qubo_config"]["exploration_seed"] = 42

    print("=== Generating Ablation Configs ===")

    generate_qubo(cfg)
    generate_qaoa(cfg)
    generate_candidate(cfg)
    generate_bias(cfg)
    generate_local(cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()