# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

from algorithms.quantum.qubo_builder import (
    QUBOConfig,
    build_qubo,
)
from algorithms.quantum.ising_mapping import qubo_to_ising
from algorithms.quantum.qaoa_solver import (
    QAOASolverConfig,
    QAOASolverResult,
    solve_qaoa,
)
from algorithms.subproblem.router import evaluate_subproblem


@dataclass
class QuantumMasterConfig:
    """
    量子主问题配置。

    round_digits:
        调用 evaluate_subproblem 时输出保留的小数位数。

    candidate_top_k:
        从 QAOA 求解器返回的 top_samples 中取前多少个候选 x，
        再用真实子问题目标 Z 做筛选。

    include_solver_best:
        是否额外把 QAOA solver 的 best_x 强制加入候选集合。
        一般建议为 True。

    deduplicate_candidates:
        是否对候选 bitstring 去重。
    """
    qubo_config: QUBOConfig | None = None
    qaoa_config: QAOASolverConfig | None = None

    round_digits: int = 6
    candidate_top_k: int = 10
    include_solver_best: bool = True
    deduplicate_candidates: bool = True


def _bitstring_to_x(bitstring: str) -> np.ndarray:
    return np.array([float(int(b)) for b in bitstring], dtype=float)


def _deduplicate_candidate_dicts(candidates: List[dict]) -> List[dict]:
    """
    对候选样本按 bitstring 去重。
    """
    seen = set()
    result = []
    for item in candidates:
        bitstring = item["bitstring"]
        if bitstring in seen:
            continue
        seen.add(bitstring)
        result.append(item)
    return result


def _collect_qaoa_candidates(
    solver_result: QAOASolverResult,
    top_k: int,
    include_solver_best: bool = True,
    deduplicate: bool = True,
) -> List[dict]:
    """
    从 QAOA 求解结果中整理候选 x。
    返回元素格式至少包含：
        {
            "bitstring": ...,
            "x": np.ndarray,
            "energy": ...,
            ...
        }
    """
    candidates: List[dict] = []

    if include_solver_best:
        candidates.append(
            {
                "bitstring": solver_result.best_bitstring,
                "x": solver_result.best_x.copy(),
                "energy": float(solver_result.best_energy),
                "probability": None,
                "gammas": solver_result.best_gammas.copy(),
                "betas": solver_result.best_betas.copy(),
                "source": "solver_best",
            }
        )

    for sample in solver_result.top_samples[:top_k]:
        item = dict(sample)
        item["x"] = np.asarray(item["x"], dtype=float).copy()
        item["source"] = "top_samples"
        candidates.append(item)

    if deduplicate:
        candidates = _deduplicate_candidate_dicts(candidates)

    return candidates


def search_x_quantum(
    problem_dict: dict,
    config: QuantumMasterConfig | None = None,
    incumbent: np.ndarray | None = None,
) -> Dict[str, Any]:
    """
    量子版 master 主入口。

    步骤：
    1) problem_dict -> QUBO
    2) QUBO -> Ising
    3) solve_qaoa 得到候选 bitstring / x
    4) 对候选 x 调用真实子问题 evaluate_subproblem
    5) 用真实目标 Z 选出最优

    返回格式与现有 classical master 尽量兼容，并补充量子信息。
    """
    if config is None:
        config = QuantumMasterConfig()

    qubo_config = config.qubo_config or QUBOConfig()
    qaoa_config = config.qaoa_config or QAOASolverConfig()

    # 若给了 incumbent 且 QUBOConfig 中启用了 hamming_penalty，
    # build_qubo 会自动利用它；否则只是不使用该信息。
    Q = build_qubo(
        problem_dict=problem_dict,
        config=qubo_config,
        incumbent=incumbent,
    )

    ising_model = qubo_to_ising(Q)

    solver_result = solve_qaoa(
        model=ising_model,
        config=qaoa_config,
    )

    qaoa_candidates = _collect_qaoa_candidates(
        solver_result=solver_result,
        top_k=config.candidate_top_k,
        include_solver_best=config.include_solver_best,
        deduplicate=config.deduplicate_candidates,
    )

    best_x: np.ndarray | None = None
    best_result: dict | None = None
    best_objective = -float("inf")

    evaluated_candidates: List[dict] = []
    failed_candidates: List[dict] = []

    for cand in qaoa_candidates:
        x = np.asarray(cand["x"], dtype=float)

        try:
            result = evaluate_subproblem(
                problem_dict,
                x,
                round_digits=config.round_digits,
            )
            z = float(result["Z"])

            record = {
                "bitstring": cand["bitstring"],
                "x": x.copy(),
                "energy": cand.get("energy"),
                "probability": cand.get("probability"),
                "gammas": cand.get("gammas"),
                "betas": cand.get("betas"),
                "source": cand.get("source"),
                "subproblem_result": result,
                "objective_value": z,
            }
            evaluated_candidates.append(record)

            if z > best_objective:
                best_objective = z
                best_x = x.copy()
                best_result = result

        except Exception as e:
            failed_candidates.append(
                {
                    "bitstring": cand["bitstring"],
                    "x": x.copy(),
                    "energy": cand.get("energy"),
                    "probability": cand.get("probability"),
                    "gammas": cand.get("gammas"),
                    "betas": cand.get("betas"),
                    "source": cand.get("source"),
                    "error": repr(e),
                }
            )
            continue

    if best_x is None or best_result is None:
        raise RuntimeError(
            "量子 master 未能从任何候选 x 中得到有效子问题结果。"
        )

    evaluated_candidates.sort(key=lambda item: item["objective_value"], reverse=True)

    return {
        "best_x": best_x,
        "best_result": best_result,
        "best_objective": best_objective,
        "evaluated_candidates": evaluated_candidates,
        "failed_candidates": failed_candidates,
        "qubo_matrix": Q,
        "ising_model": ising_model,
        "qaoa_solver_result": solver_result,
        "extra": {
            "num_qaoa_candidates": len(qaoa_candidates),
            "num_evaluated_candidates": len(evaluated_candidates),
            "num_failed_candidates": len(failed_candidates),
            "qaoa_best_bitstring": solver_result.best_bitstring,
            "qaoa_best_energy": solver_result.best_energy,
            "qaoa_best_gammas": solver_result.best_gammas.copy(),
            "qaoa_best_betas": solver_result.best_betas.copy(),
        },
    }


if __name__ == "__main__":
    import json

    # 本地自检：用真实 problem_dict 跑一遍量子 master
    with open("../../data/raw/problem_micp_1.json", "r", encoding="utf-8") as f:
        problem_dict = json.load(f)

    config = QuantumMasterConfig(
        qubo_config=QUBOConfig(
            objective_scale=1.0,
            resource_penalty=10.0,
            hamming_penalty=0.0,
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
    )

    result = search_x_quantum(problem_dict, config=config)

    print("=== Quantum Master Best ===")
    print("best_objective:", result["best_objective"])
    print("best_x:", result["best_x"].tolist())
    print("best_result:", result["best_result"])

    print("\n=== Top Evaluated Candidates ===")
    for item in result["evaluated_candidates"][:5]:
        print(
            {
                "bitstring": item["bitstring"],
                "objective_value": round(item["objective_value"], 6),
                "energy": None if item["energy"] is None else round(item["energy"], 6),
                "probability": item["probability"],
            }
        )