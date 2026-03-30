import json
import math
import pathlib
import sys

import numpy as np
import pytest

# 让 tests/ 可以直接找到项目根目录
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from algorithms.subproblem.micp_qp import evaluate_x, ProblemData, solve_qp_given_x
from algorithms.subproblem.micp_qp.feasibility import check_primal_feasibility

from algorithms.subproblem.micp_qp.objectives import (
    total_objective_value,
    remaining_resource,
)


DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXAMPLES_DIR = DATA_DIR / "examples"


def load_json(path: pathlib.Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_list_close(actual, expected, tol=1e-2):
    assert len(actual) == len(expected), f"长度不一致: {len(actual)} != {len(expected)}"
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert math.isclose(a, e, rel_tol=tol, abs_tol=tol), (
            f"第 {i} 个元素不一致: actual={a}, expected={e}"
        )


def test_example_answer_match():
    """
    用官方 example_problem + example_answer 做回归测试。
    这是最重要的一条测试。
    """
    problem = load_json(EXAMPLES_DIR / "example_problem.json")
    answer = load_json(EXAMPLES_DIR / "example_answer.json")

    x = np.array(answer["x"], dtype=float)
    result = evaluate_x(problem, x, round_digits=2)

    assert result["x"] == answer["x"]
    assert_list_close(result["y"], answer["y"], tol=1e-2)
    assert math.isclose(result["Z"], answer["Z"], rel_tol=1e-2, abs_tol=1e-2)
    assert_list_close(result["r"], answer["r"], tol=1e-2)


def test_zero_x_returns_zero_y():
    """
    队友代码里专门处理了 x 全 0 的情形，这里单独验证。
    """
    problem_dict = load_json(RAW_DIR / "problem_micp_1.json")
    problem = ProblemData.from_dict(problem_dict)

    x = np.zeros(problem.n, dtype=float)
    result = solve_qp_given_x(problem, x)

    assert np.allclose(result.y, 0.0)
    assert result.converged is True
    assert result.primal_violation <= 1e-8
    assert np.allclose(result.remaining_resource, problem.resource_limit)


def test_micp_outputs_are_feasible():
    """
    对 MICP 实例给一个合法 x，检查输出 y 是否满足原始约束。
    """
    problem_dict = load_json(RAW_DIR / "problem_micp_1.json")
    problem = ProblemData.from_dict(problem_dict)

    # 给一个简单合法的二进制向量
    x = np.ones(problem.n, dtype=float)

    result_dict, solver_result = evaluate_x(
        problem_dict,
        x,
        round_digits=6,
        return_solver_result=True,
    )

    report = solver_result.feasibility_report
    assert report["feasible"], f"可行性失败: {report}"
    assert report["lower_violation"] <= 1e-6
    assert report["upper_violation"] <= 1e-6
    assert report["resource_violation"] <= 1e-6

    # 输出格式也顺便检查一下
    assert "x" in result_dict
    assert "y" in result_dict
    assert "Z" in result_dict
    assert "r" in result_dict
    assert len(result_dict["x"]) == problem.n
    assert len(result_dict["y"]) == problem.n
    assert len(result_dict["r"]) == problem.m


def test_multiple_micp_instances_run():
    """
    确认多个 MICP 样例都能正常跑通，不报错，并返回有限数值。
    """
    filenames = [
        "problem_micp_1.json",
        "problem_micp_2.json",
    ]

    for name in filenames:
        problem_dict = load_json(RAW_DIR / name)
        problem = ProblemData.from_dict(problem_dict)

        # 构造一个简单的候选 x：奇数位置开，偶数位置关
        x = np.array([(i % 2) for i in range(problem.n)], dtype=float)

        result = evaluate_x(problem_dict, x, round_digits=6)

        assert len(result["x"]) == problem.n
        assert len(result["y"]) == problem.n
        assert len(result["r"]) == problem.m
        assert math.isfinite(result["Z"])

        # 再做一次显式可行性检查
        y = np.array(result["y"], dtype=float)
        report = check_primal_feasibility(problem, x, y, tol=1e-5)
        assert report["feasible"], f"{name} 可行性失败: {report}"


def test_invalid_x_raises():
    """
    非法 x 应该抛 ValueError。
    """
    problem_dict = load_json(RAW_DIR / "problem_micp_1.json")
    bad_x = np.array([0.2, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=float)

    try:
        evaluate_x(problem_dict, bad_x)
        raise AssertionError("非法 x 没有抛异常")
    except ValueError:
        pass

def test_random_x_batch_feasibility_micp():
    """
    对 MICP 实例随机采样多个合法 x，检查是否都能跑通且输出可行。
    """
    rng = np.random.default_rng(42)
    filenames = ["problem_micp_1.json", "problem_micp_2.json"]

    for name in filenames:
        problem_dict = load_json(RAW_DIR / name)
        problem = ProblemData.from_dict(problem_dict)

        for _ in range(30):
            x = rng.integers(0, 2, size=problem.n).astype(float)
            result_dict = evaluate_x(problem_dict, x, round_digits=8)

            y = np.array(result_dict["y"], dtype=float)
            report = check_primal_feasibility(problem, x, y, tol=1e-5)

            assert report["feasible"], f"{name} 随机 x 不可行: x={x}, report={report}"


def test_objective_and_resource_consistency_micp():
    """
    检查：
    1. result['Z'] 是否与 total_objective_value 自洽
    2. result['r'] 是否等于 remaining_resource
    """
    problem_dict = load_json(RAW_DIR / "problem_micp_1.json")
    problem = ProblemData.from_dict(problem_dict)

    x = np.array([1 if i % 2 == 0 else 0 for i in range(problem.n)], dtype=float)
    result_dict = evaluate_x(problem_dict, x, round_digits=8)

    y = np.array(result_dict["y"], dtype=float)
    z_expected = total_objective_value(problem, x, y)
    r_expected = remaining_resource(problem, y)

    assert math.isclose(result_dict["Z"], z_expected, rel_tol=1e-6, abs_tol=1e-6)
    assert np.allclose(np.array(result_dict["r"], dtype=float), r_expected, atol=1e-6)


def test_reject_zero_beta_example_for_micp_solver():
    """
    MICP QP 子问题求解器应拒绝 beta 全 0 的 example/MILP 数据。
    """
    problem_dict = load_json(EXAMPLES_DIR / "example_problem.json")
    x = np.array([0, 1, 1, 1, 1], dtype=float)

    with pytest.raises(ValueError, match="beta_i > 0"):
        evaluate_x(problem_dict, x)


def test_invalid_x_length_raises_micp():
    """
    x 维度不对时应抛异常。
    """
    problem_dict = load_json(RAW_DIR / "problem_micp_1.json")
    n = problem_dict["product_count"]

    bad_x = np.ones(n + 2, dtype=float)

    with pytest.raises(ValueError, match="shape"):
        evaluate_x(problem_dict, bad_x)


def test_missing_field_raises_micp():
    """
    输入字段缺失时应抛 KeyError。
    """
    problem_dict = load_json(RAW_DIR / "problem_micp_1.json")
    broken = dict(problem_dict)
    broken.pop("alpha")

    x = np.ones(problem_dict["product_count"], dtype=float)

    with pytest.raises(KeyError):
        evaluate_x(broken, x)


def test_negative_resource_limit_raises_micp():
    """
    非法资源上限应被拒绝。
    """
    problem_dict = load_json(RAW_DIR / "problem_micp_1.json")
    broken = dict(problem_dict)
    broken["resource_limit"] = broken["resource_limit"].copy()
    broken["resource_limit"][0] = -5

    x = np.ones(problem_dict["product_count"], dtype=float)

    with pytest.raises(ValueError, match="resource_limit"):
        evaluate_x(broken, x)