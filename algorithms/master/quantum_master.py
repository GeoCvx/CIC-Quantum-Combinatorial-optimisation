# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

from algorithms.quantum.qubo_builder import QUBOConfig, build_qubo
from algorithms.quantum.ising_mapping import qubo_to_ising
from algorithms.quantum.qaoa_solver import QAOASolverConfig, QAOASolverResult, solve_qaoa
from algorithms.subproblem.router import evaluate_subproblem
from algorithms.preprocess.feature_builder import ProblemStats
from algorithms.feedback.bias_updater import BiasState


@dataclass
class QuantumMasterConfig:
    qubo_config: QUBOConfig | None = None
    qaoa_config: QAOASolverConfig | None = None
    round_digits: int = 6

    candidate_top_k: int = 10
    include_solver_best: bool = True
    deduplicate_candidates: bool = True

    # 新增：扩展候选
    add_mutations_from_best: bool = True
    add_mutations_from_incumbent: bool = True
    max_best_mutations: int = 3
    max_incumbent_mutations: int = 3


def _bitstring_from_x(x: np.ndarray) -> str:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return "".join(str(int(round(v))) for v in arr)


def _deduplicate_candidate_dicts(candidates: List[dict]) -> List[dict]:
    seen = set()
    result = []
    for item in candidates:
        bitstring = item.get("bitstring") or _bitstring_from_x(item["x"])
        if bitstring in seen:
            continue
        seen.add(bitstring)
        item = dict(item)
        item["bitstring"] = bitstring
        result.append(item)
    return result


def _make_single_flip_mutations(x: np.ndarray, max_count: int, source: str) -> list[dict]:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(x)
    candidates = []
    for i in range(n):
        x_new = x.copy()
        x_new[i] = 1.0 - x_new[i]
        candidates.append(
            {
                "bitstring": _bitstring_from_x(x_new),
                "x": x_new.copy(),
                "energy": None,
                "probability": None,
                "gammas": None,
                "betas": None,
                "source": source,
            }
        )
        if len(candidates) >= max_count:
            break
    return candidates


def _collect_qaoa_candidates(
    solver_result: QAOASolverResult,
    top_k: int,
    include_solver_best: bool = True,
    deduplicate: bool = True,
    incumbent: np.ndarray | None = None,
    add_mutations_from_best: bool = True,
    add_mutations_from_incumbent: bool = True,
    max_best_mutations: int = 3,
    max_incumbent_mutations: int = 3,
) -> List[dict]:
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
        item["bitstring"] = item.get("bitstring") or _bitstring_from_x(item["x"])
        candidates.append(item)

    # 新增：对 best 做单点扰动
    if add_mutations_from_best:
        candidates.extend(
            _make_single_flip_mutations(
                solver_result.best_x,
                max_count=max_best_mutations,
                source="mutated_from_best",
            )
        )

    # 新增：对 incumbent 做单点扰动
    if add_mutations_from_incumbent and incumbent is not None:
        candidates.extend(
            _make_single_flip_mutations(
                np.asarray(incumbent, dtype=float).reshape(-1),
                max_count=max_incumbent_mutations,
                source="mutated_from_incumbent",
            )
        )

    if deduplicate:
        candidates = _deduplicate_candidate_dicts(candidates)

    return candidates


def search_x_quantum(
    problem_dict: dict,
    config: QuantumMasterConfig | None = None,
    incumbent: np.ndarray | None = None,
    stats: ProblemStats | None = None,
    bias_state: BiasState | None = None,
) -> Dict[str, Any]:
    if config is None:
        config = QuantumMasterConfig()

    qubo_config = config.qubo_config or QUBOConfig()
    qaoa_config = config.qaoa_config or QAOASolverConfig()

    Q = build_qubo(
        problem_dict=problem_dict,
        config=qubo_config,
        incumbent=incumbent,
        stats=stats,
        bias_state=bias_state,
    )

    ising_model = qubo_to_ising(Q)
    solver_result = solve_qaoa(model=ising_model, config=qaoa_config)

    qaoa_candidates = _collect_qaoa_candidates(
        solver_result=solver_result,
        top_k=config.candidate_top_k,
        include_solver_best=config.include_solver_best,
        deduplicate=config.deduplicate_candidates,
        incumbent=incumbent,
        add_mutations_from_best=config.add_mutations_from_best,
        add_mutations_from_incumbent=config.add_mutations_from_incumbent,
        max_best_mutations=config.max_best_mutations,
        max_incumbent_mutations=config.max_incumbent_mutations,
    )

    best_x: np.ndarray | None = None
    best_result: dict | None = None
    best_objective = -float("inf")

    evaluated_candidates: List[dict] = []
    failed_candidates: List[dict] = []

    for cand in qaoa_candidates:
        x = np.asarray(cand["x"], dtype=float).reshape(-1)
        try:
            result = evaluate_subproblem(problem_dict, x, round_digits=config.round_digits)
            z = float(result["Z"])

            record = {
                "bitstring": cand.get("bitstring") or _bitstring_from_x(x),
                "x": x.copy(),
                "energy": cand.get("energy"),
                "probability": cand.get("probability"),
                "gammas": cand.get("gammas"),
                "betas": cand.get("betas"),
                "source": cand.get("source"),
                "subproblem_result": result,
                "objective_value": z,
                "feasible": True,
            }
            evaluated_candidates.append(record)

            if z > best_objective:
                best_objective = z
                best_x = x.copy()
                best_result = result

        except Exception as e:
            failed_candidates.append(
                {
                    "bitstring": cand.get("bitstring") or _bitstring_from_x(x),
                    "x": x.copy(),
                    "energy": cand.get("energy"),
                    "probability": cand.get("probability"),
                    "source": cand.get("source"),
                    "error": repr(e),
                    "feasible": False,
                }
            )

    if best_x is None or best_result is None:
        raise RuntimeError("量子 master 未能从任何候选 x 中得到有效子问题结果。")

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
        },
    }