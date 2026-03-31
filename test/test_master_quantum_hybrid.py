import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from algorithms.master.quantum_master import (
    QuantumMasterConfig,
    search_x_quantum,
)
from algorithms.master.hybrid_master import (
    HybridMasterConfig,
    search_x_hybrid,
)
from algorithms.quantum.qubo_builder import QUBOConfig
from algorithms.quantum.qaoa_solver import QAOASolverConfig


DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


def load_json(path: pathlib.Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
            per_run_top_k=3,
            final_top_k=5,
        ),
        round_digits=6,
        candidate_top_k=3,
        include_solver_best=True,
        deduplicate_candidates=True,
    )


def make_hybrid_config() -> HybridMasterConfig:
    return HybridMasterConfig(
        use_classical=True,
        use_quantum=True,
        classical_num_starts=5,
        classical_local_iter=20,
        classical_seed=42,
        quantum_config=QuantumMasterConfig(
            qubo_config=QUBOConfig(
                objective_scale=1.0,
                resource_penalty=10.0,
                hamming_penalty=2.0,
                demand_weight=1.0,
            ),
            qaoa_config=QAOASolverConfig(
                p=1,
                shots=128,
                gamma_values=np.array([0.2, 0.5], dtype=float),
                beta_values=np.array([0.2, 0.5], dtype=float),
                seed=42,
                per_run_top_k=3,
                final_top_k=5,
            ),
            round_digits=6,
            candidate_top_k=3,
            include_solver_best=True,
            deduplicate_candidates=True,
        ),
        fallback_to_classical=True,
    )


def assert_binary_vector(x, n: int):
    arr = np.asarray(x, dtype=float).reshape(-1)
    assert arr.shape == (n,)
    assert np.all((arr == 0.0) | (arr == 1.0)), f"x 不是合法 0/1 向量: {arr}"


def test_search_x_quantum_runs_on_micp():
    problem_dict = load_json(RAW_DIR / "problem_micp_1.json")
    n = problem_dict["product_count"]

    result = search_x_quantum(
        problem_dict,
        config=make_quantum_config(),
    )

    assert "best_x" in result
    assert "best_result" in result
    assert "best_objective" in result
    assert "evaluated_candidates" in result
    assert "failed_candidates" in result
    assert "qaoa_solver_result" in result
    assert "extra" in result

    assert_binary_vector(result["best_x"], n)
    assert isinstance(result["best_objective"], (int, float))
    assert result["best_result"]["x"] == [int(v) for v in result["best_x"]]

    best_result = result["best_result"]
    assert "y" in best_result
    assert "Z" in best_result
    assert "r" in best_result


def test_search_x_quantum_with_incumbent_runs():
    problem_dict = load_json(RAW_DIR / "problem_micp_1.json")
    n = problem_dict["product_count"]
    incumbent = np.zeros(n, dtype=float)
    incumbent[0] = 1.0

    cfg = make_quantum_config()
    cfg.qubo_config.hamming_penalty = 2.0

    result = search_x_quantum(
        problem_dict,
        config=cfg,
        incumbent=incumbent,
    )

    assert_binary_vector(result["best_x"], n)
    assert isinstance(result["best_objective"], (int, float))


def test_search_x_hybrid_runs_on_micp():
    problem_dict = load_json(RAW_DIR / "problem_micp_1.json")
    n = problem_dict["product_count"]

    result = search_x_hybrid(
        problem_dict,
        config=make_hybrid_config(),
    )

    assert "best_x" in result
    assert "best_result" in result
    assert "best_objective" in result
    assert "selected_master" in result
    assert "classical_result" in result
    assert "quantum_result" in result
    assert "extra" in result

    assert result["selected_master"] in {"classical", "quantum"}
    assert_binary_vector(result["best_x"], n)
    assert isinstance(result["best_objective"], (int, float))


def test_hybrid_best_objective_matches_selected_master():
    problem_dict = load_json(RAW_DIR / "problem_micp_1.json")

    result = search_x_hybrid(
        problem_dict,
        config=make_hybrid_config(),
    )

    selected = result["selected_master"]
    if selected == "classical":
        assert result["classical_result"] is not None
        assert result["best_objective"] == result["classical_result"]["best_objective"]
    else:
        assert result["quantum_result"] is not None
        assert result["best_objective"] == result["quantum_result"]["best_objective"]


def test_hybrid_fallback_to_classical_when_quantum_disabled():
    problem_dict = load_json(RAW_DIR / "problem_micp_1.json")

    cfg = make_hybrid_config()
    cfg.use_quantum = False

    result = search_x_hybrid(problem_dict, config=cfg)

    assert result["selected_master"] == "classical"
    assert result["classical_result"] is not None
    assert result["quantum_result"] is None


# def test_search_x_quantum_invalid_candidate_top_k_raises():
#     problem_dict = load_json(RAW_DIR / "problem_micp_1.json")
#
#     cfg = make_quantum_config()
#     cfg.candidate_top_k = 0
#
#     with pytest.raises(Exception):
#         search_x_quantum(problem_dict, config=cfg)