# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from .problem_data import ProblemData
from .lp_solver import LPSolverConfig, LPSolverResult, solve_lp_given_x
from .feasibility import validate_x_binary


def evaluate_x(
    problem_dict: dict,
    x: np.ndarray,
    round_digits: int = 2,
    solver_config: LPSolverConfig | None = None,
    return_solver_result: bool = False,
) -> dict | tuple[dict, LPSolverResult]:
    """
    LP_solver 对外统一接口。

    参数
    ----
    problem_dict : dict
        从 json.load(...) 得到的比赛输入字典

    x : np.ndarray
        固定的 0/1 向量

    round_digits : int
        输出中 y / Z / r 保留的小数位数

    solver_config : LPSolverConfig | None
        LP 求解器配置；若为 None，则使用默认配置

    return_solver_result : bool
        是否额外返回 LP 求解器原始结果对象

    返回
    ----
    result_dict 或 (result_dict, lp_result)

    result_dict 格式：
    {
        "x": [...],
        "y": [...],
        "Z": ...,
        "r": [...]
    }
    """
    problem = ProblemData.from_dict(problem_dict)
    x = validate_x_binary(x, problem.n)

    if solver_config is None:
        solver_config = LPSolverConfig()

    lp_result = solve_lp_given_x(problem, x, config=solver_config)

    result_dict = {
        "x": x.astype(int).tolist(),
        "y": np.round(lp_result.y, round_digits).tolist(),
        "Z": round(float(lp_result.total_objective), round_digits),
        "r": np.round(lp_result.remaining_resource, round_digits).tolist(),
    }

    if return_solver_result:
        return result_dict, lp_result
    return result_dict