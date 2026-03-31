import time
from typing import Any

from algorithms.master.classical_master import search_x_classical
from algorithms.master.quantum_master import search_x_quantum
from algorithms.master.hybrid_master import search_x_hybrid


def _run_master(
    problem_dict: dict,
    master_mode: str = "classical",
    master_config: Any = None,
):
    """
    统一调度不同 master。
    """
    master_mode = master_mode.lower()

    if master_mode == "classical":
        if master_config is None:
            return search_x_classical(problem_dict)
        if isinstance(master_config, dict):
            return search_x_classical(problem_dict, **master_config)
        raise TypeError("classical 模式下 master_config 应为 dict 或 None。")

    if master_mode == "quantum":
        return search_x_quantum(problem_dict, config=master_config)

    if master_mode == "hybrid":
        return search_x_hybrid(problem_dict, config=master_config)

    raise ValueError(
        f"不支持的 master_mode: {master_mode}，可选为 classical / quantum / hybrid"
    )


def solve_micp(
    problem_dict: dict,
    master_mode: str = "classical",
    master_config: Any = None,
):
    start = time.time()

    search_result = _run_master(
        problem_dict=problem_dict,
        master_mode=master_mode,
        master_config=master_config,
    )

    runtime = time.time() - start

    return {
        "status": "feasible",
        "objective_value": search_result["best_objective"],
        "solution": search_result["best_result"],
        "runtime": runtime,
        "extra": {
            "problem_type": "MICP",
            "master_mode": master_mode,
            "master_summary": search_result.get("extra", {}),
        },
    }