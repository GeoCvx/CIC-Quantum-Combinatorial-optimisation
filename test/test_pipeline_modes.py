import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from algorithms.pipeline.milp_pipeline import solve_milp
from algorithms.pipeline.micp_pipeline import solve_micp
from algorithms.master.quantum_master import QuantumMasterConfig
from algorithms.master.hybrid_master import HybridMasterConfig
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


def assert_pipeline_result(result: dict, expected_problem_type: str, expected_master_mode: str):
    assert "status" in result
    assert "objective_value" in result
    assert "solution" in result
    assert "runtime" in result
    assert "extra" in result

    assert result["status"] == "feasible"
    assert isinstance(result["objective_value"], (int, float))
    assert result["runtime"] >= 0.0

    extra = result["extra"]
    assert extra["problem_type"] == expected_problem_type
    assert extra["master_mode"] == expected_master_mode

    sol = result["solution"]
    assert "x" in sol
    assert "y" in sol
    assert "Z" in sol
    assert "r" in sol


def test_solve_milp_classical_runs():
    problem_dict = load_json(RAW_DIR / "problem_milp_1.json")

    result = solve_milp(
        problem_dict,
        master_mode="classical",
        master_config={"num_starts": 5, "local_iter": 20, "seed": 42},
    )

    assert_pipeline_result(result, expected_problem_type="MILP", expected_master_mode="classical")


def test_solve_micp_classical_runs():
    problem_dict = load_json(RAW_DIR / "problem_micp_1.json")

    result = solve_micp(
        problem_dict,
        master_mode="classical",
        master_config={"num_starts": 5, "local_iter": 20, "seed": 42},
    )

    assert_pipeline_result(result, expected_problem_type="MICP", expected_master_mode="classical")


def test_solve_micp_quantum_runs():
    problem_dict = load_json(RAW_DIR / "problem_micp_1.json")

    result = solve_micp(
        problem_dict,
        master_mode="quantum",
        master_config=make_quantum_config(),
    )

    assert_pipeline_result(result, expected_problem_type="MICP", expected_master_mode="quantum")


def test_solve_micp_hybrid_runs():
    problem_dict = load_json(RAW_DIR / "problem_micp_1.json")

    result = solve_micp(
        problem_dict,
        master_mode="hybrid",
        master_config=make_hybrid_config(),
    )

    assert_pipeline_result(result, expected_problem_type="MICP", expected_master_mode="hybrid")


def test_invalid_master_mode_raises_for_milp():
    problem_dict = load_json(RAW_DIR / "problem_milp_1.json")

    with pytest.raises(ValueError, match="master_mode"):
        solve_milp(problem_dict, master_mode="invalid_mode")


def test_invalid_master_mode_raises_for_micp():
    problem_dict = load_json(RAW_DIR / "problem_micp_1.json")

    with pytest.raises(ValueError, match="master_mode"):
        solve_micp(problem_dict, master_mode="invalid_mode")


def test_classical_master_config_type_error():
    problem_dict = load_json(RAW_DIR / "problem_milp_1.json")

    with pytest.raises(TypeError):
        solve_milp(
            problem_dict,
            master_mode="classical",
            master_config=make_quantum_config(),  # 故意传错类型
        )