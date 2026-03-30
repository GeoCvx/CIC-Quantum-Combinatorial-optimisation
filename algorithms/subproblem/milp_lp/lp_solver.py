# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linprog

from .problem_data import ProblemData
from .feasibility import validate_x_binary, upper_bound_from_x, check_primal_feasibility
from .objectives import lp_objective_value, total_objective_value, remaining_resource


@dataclass
class LPSolverConfig:
    """
    scipy.optimize.linprog 配置。
    """
    method: str = "highs"
    tol: float = 1e-9
    presolve: bool = True


@dataclass
class LPSolverResult:
    y: np.ndarray
    lp_objective: float
    total_objective: float
    remaining_resource: np.ndarray
    feasibility_report: dict
    success: bool
    status: int
    message: str
    nit: int | None = None
    slack: np.ndarray | None = None
    upper_dual: np.ndarray | None = None
    resource_dual: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


def solve_lp_given_x(
    problem: ProblemData,
    x: np.ndarray,
    config: LPSolverConfig | None = None,
) -> LPSolverResult:
    """
    固定 x 后求解连续 LP 子问题：

        max c^T y
        s.t. A y <= R
             0 <= y <= D*x

    其中 c = price - alpha
    """
    if config is None:
        config = LPSolverConfig()

    problem.validate_lp_compatible()
    x = validate_x_binary(x, problem.n)

    c = problem.c
    A = problem.consumption_matrix
    R = problem.resource_limit
    u = upper_bound_from_x(problem, x)

    # 若 x 全 0，则 y 必为 0
    if np.all(u <= 1e-15):
        y = np.zeros(problem.n, dtype=float)
        return LPSolverResult(
            y=y,
            lp_objective=lp_objective_value(problem, y),
            total_objective=total_objective_value(problem, x, y),
            remaining_resource=remaining_resource(problem, y),
            feasibility_report=check_primal_feasibility(problem, x, y, tol=config.tol),
            success=True,
            status=0,
            message="x 全 0，直接返回零解。",
            nit=0,
            slack=R.copy(),
            upper_dual=np.zeros(problem.n, dtype=float),
            resource_dual=np.zeros(problem.m, dtype=float),
        )

    # linprog 默认做最小化，因此取 -c
    res = linprog(
        c=-c,
        A_ub=A,
        b_ub=R,
        bounds=[(0.0, float(ui)) for ui in u],
        method=config.method,
        options={"presolve": config.presolve},
    )

    if not res.success:
        raise RuntimeError(
            f"LP 子问题求解失败: status={res.status}, message={res.message}"
        )

    y = np.asarray(res.x, dtype=float)
    feas = check_primal_feasibility(problem, x, y, tol=max(1e-7, 100.0 * config.tol))

    # HiGHS 一般会返回下列对偶信息；做兼容式读取
    upper_dual = None
    resource_dual = None
    try:
        # 对 A_ub y <= R 的边际值
        resource_dual = np.asarray(res.ineqlin.marginals, dtype=float)
    except Exception:
        pass

    try:
        # upper bound 的边际值
        upper_dual = np.asarray(res.upper.marginals, dtype=float)
    except Exception:
        pass

    return LPSolverResult(
        y=y,
        lp_objective=lp_objective_value(problem, y),
        total_objective=total_objective_value(problem, x, y),
        remaining_resource=remaining_resource(problem, y),
        feasibility_report=feas,
        success=bool(res.success),
        status=int(res.status),
        message=str(res.message),
        nit=getattr(res, "nit", None),
        slack=np.asarray(getattr(res, "slack", np.array([])), dtype=float),
        upper_dual=upper_dual,
        resource_dual=resource_dual,
        extra={"fun_min": float(res.fun)},
    )