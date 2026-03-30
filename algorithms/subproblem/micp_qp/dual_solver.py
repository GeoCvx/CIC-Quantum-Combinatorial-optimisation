# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .problem_data import ProblemData
from .objectives import qp_objective_value, total_objective_value, remaining_resource
from .feasibility import (
    validate_x_binary,
    upper_bound_from_x,
    check_primal_feasibility,
    repair_y_numerically,
)


@dataclass
class QPSolverConfig:
    """
    QP 求解器参数。
    """
    max_iter: int = 5000
    tol: float = 1e-8

    # 对偶空间投影梯度的初始步长
    step_init: float = 1.0

    # 回溯线搜索参数
    backtrack_factor: float = 0.5
    min_step: float = 1e-12

    # 是否对最终 y 做数值修复
    repair_solution: bool = True

    # 是否保存迭代历史
    store_history: bool = False
    history_every: int = 20


@dataclass
class QPSolverResult:
    """
    固定 x 后 QP 子问题的求解结果。
    """
    y: np.ndarray
    lambda_: np.ndarray

    qp_objective: float
    total_objective: float
    dual_objective: float

    converged: bool
    iterations: int

    primal_violation: float
    pg_norm: float

    remaining_resource: np.ndarray
    feasibility_report: dict

    history: list[dict] = field(default_factory=list)


def recover_y_from_lambda(
    lambda_: np.ndarray,
    c: np.ndarray,
    beta: np.ndarray,
    A: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """
    给定对偶变量 lambda，恢复原始变量 y(lambda)：
        y_i = clip((c_i - a_i^T lambda) / (2 beta_i), 0, u_i)
    """
    shifted_linear = c - A.T @ lambda_
    y = shifted_linear / (2.0 * beta)
    y = np.clip(y, 0.0, u)
    return y


def dual_objective_and_gradient(
    lambda_: np.ndarray,
    c: np.ndarray,
    beta: np.ndarray,
    A: np.ndarray,
    R: np.ndarray,
    u: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    返回：
    - 对偶目标 g(lambda)
    - 梯度 grad = R - A y(lambda)
    - 当前 y(lambda)
    """
    y = recover_y_from_lambda(lambda_, c, beta, A, u)
    shifted_linear = c - A.T @ lambda_

    # phi_i(lambda) = max_{0<=y_i<=u_i} [ shifted_linear_i * y_i - beta_i y_i^2 ]
    phi = shifted_linear * y - beta * y * y
    g = float(np.dot(lambda_, R) + np.sum(phi))

    grad = R - A @ y
    return g, grad, y


def projected_gradient_mapping_norm(lambda_: np.ndarray, grad: np.ndarray) -> float:
    """
    投影梯度映射的无穷范数，用于判断收敛。
    """
    projected = np.maximum(lambda_ - grad, 0.0)
    return float(np.max(np.abs(lambda_ - projected)))


def solve_qp_given_x(
    problem: ProblemData,
    x: np.ndarray,
    config: QPSolverConfig | None = None,
    lambda0: np.ndarray | None = None,
) -> QPSolverResult:
    """
    对外唯一核心接口：
        固定 x 后求解连续 QP 子问题。

    参数
    ----
    problem : ProblemData
    x       : 固定的 0/1 向量
    config  : 求解器配置
    lambda0 : 对偶变量初值

    返回
    ----
    QPSolverResult
    """
    if config is None:
        config = QPSolverConfig()

    problem.validate_qp_compatible()
    x = validate_x_binary(x, problem.n)

    c = problem.c
    beta = problem.beta
    A = problem.consumption_matrix
    R = problem.resource_limit
    u = upper_bound_from_x(problem, x)

    # 若 x 全 0，则 y 必为 0
    if np.all(u <= 1e-15):
        y = np.zeros(problem.n, dtype=float)
        return QPSolverResult(
            y=y,
            lambda_=np.zeros(problem.m, dtype=float),
            qp_objective=qp_objective_value(problem, y),
            total_objective=total_objective_value(problem, x, y),
            dual_objective=0.0,
            converged=True,
            iterations=0,
            primal_violation=0.0,
            pg_norm=0.0,
            remaining_resource=remaining_resource(problem, y),
            feasibility_report=check_primal_feasibility(problem, x, y, tol=config.tol),
            history=[],
        )

    if lambda0 is None:
        lambda_ = np.zeros(problem.m, dtype=float)
    else:
        lambda_ = np.asarray(lambda0, dtype=float).reshape(-1)
        if lambda_.shape != (problem.m,):
            raise ValueError(f"lambda0 shape 应为 ({problem.m},)，实际为 {lambda_.shape}")
        lambda_ = np.maximum(lambda_, 0.0)

    g, grad, y = dual_objective_and_gradient(lambda_, c, beta, A, R, u)

    converged = False
    history: list[dict] = []
    iterations = 0

    for it in range(1, config.max_iter + 1):
        iterations = it

        usage = A @ y
        primal_violation = max(0.0, float(np.max(usage - R)))
        pg_norm = projected_gradient_mapping_norm(lambda_, grad)

        if config.store_history and (it == 1 or it % config.history_every == 0):
            history.append(
                {
                    "iter": it,
                    "dual_objective": g,
                    "pg_norm": pg_norm,
                    "primal_violation": primal_violation,
                }
            )

        # 收敛判据：投影梯度足够小 + 原始资源违规足够小
        if pg_norm <= config.tol and primal_violation <= 10.0 * config.tol:
            converged = True
            break

        # 对偶空间投影梯度 + 回溯线搜索
        step = config.step_init
        accepted = False

        while step >= config.min_step:
            candidate_lambda = np.maximum(lambda_ - step * grad, 0.0)
            candidate_g, candidate_grad, candidate_y = dual_objective_and_gradient(
                candidate_lambda, c, beta, A, R, u
            )

            # 对偶目标下降则接受
            if candidate_g <= g + 1e-14:
                accepted = True
                break

            step *= config.backtrack_factor

        if not accepted:
            candidate_lambda = np.maximum(lambda_ - config.min_step * grad, 0.0)
            candidate_g, candidate_grad, candidate_y = dual_objective_and_gradient(
                candidate_lambda, c, beta, A, R, u
            )

        lambda_ = candidate_lambda
        g = candidate_g
        grad = candidate_grad
        y = candidate_y

    # 结束后做数值修复，增强最终可行性
    if config.repair_solution:
        y = repair_y_numerically(problem, x, y, tol=100.0 * config.tol)

    feasibility_report = check_primal_feasibility(problem, x, y, tol=100.0 * config.tol)
    final_usage = A @ y
    primal_violation = max(0.0, float(np.max(final_usage - R)))
    pg_norm = projected_gradient_mapping_norm(lambda_, grad)

    return QPSolverResult(
        y=y,
        lambda_=lambda_,
        qp_objective=qp_objective_value(problem, y),
        total_objective=total_objective_value(problem, x, y),
        dual_objective=g,
        converged=converged,
        iterations=iterations,
        primal_violation=primal_violation,
        pg_norm=pg_norm,
        remaining_resource=remaining_resource(problem, y),
        feasibility_report=feasibility_report,
        history=history,
    )