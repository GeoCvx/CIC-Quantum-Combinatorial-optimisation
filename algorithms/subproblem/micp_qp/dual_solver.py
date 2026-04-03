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

    # 平滑情形（全部 beta_i > 0）下，对偶空间投影梯度的初始步长
    step_init: float = 1.0

    # 平滑情形下，回溯线搜索参数
    backtrack_factor: float = 0.5
    min_step: float = 1e-12

    # 是否对最终 y 做数值修复
    repair_solution: bool = True

    # 是否保存迭代历史
    store_history: bool = False
    history_every: int = 20

    # beta 是否视为 0 的阈值
    zero_beta_eps: float = 1e-12

    # 混合情形（存在 beta_i = 0）时，投影次梯度法的步长缩放系数
    subgradient_step_scale: float = 1.0


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
    eps: float = 1e-12,
) -> np.ndarray:
    """
    给定对偶变量 lambda，恢复原始变量 y(lambda)。

    分两类处理：

    1) beta_i > eps：
       y_i = clip((c_i - a_i^T lambda) / (2 beta_i), 0, u_i)

    2) beta_i <= eps：
       内层退化为线性问题：
           max_{0<=y_i<=u_i} (c_i - a_i^T lambda) y_i
       因此：
       - 若系数 > 0，取 u_i
       - 若系数 < 0，取 0
       - 若系数 = 0，任意可行值都可；这里统一取 0
    """
    shifted_linear = c - A.T @ lambda_
    y = np.zeros_like(c, dtype=float)

    quad_mask = beta > eps
    lin_mask = ~quad_mask

    if np.any(quad_mask):
        y_quad = shifted_linear[quad_mask] / (2.0 * beta[quad_mask])
        y[quad_mask] = np.clip(y_quad, 0.0, u[quad_mask])

    if np.any(lin_mask):
        y_lin = np.zeros(np.sum(lin_mask), dtype=float)
        coeff_lin = shifted_linear[lin_mask]
        u_lin = u[lin_mask]

        # 线性项系数为正，则取上界
        y_lin[coeff_lin > eps] = u_lin[coeff_lin > eps]

        # 系数 <= eps 时，统一取 0
        y[lin_mask] = y_lin

    return y


def dual_objective_and_gradient(
    lambda_: np.ndarray,
    c: np.ndarray,
    beta: np.ndarray,
    A: np.ndarray,
    R: np.ndarray,
    u: np.ndarray,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    返回：
    - 对偶目标 g(lambda)
    - grad = R - A y(lambda)
    - 当前 y(lambda)

    注意：
    - 当所有 beta_i > 0 时，grad 是梯度
    - 当存在 beta_i = 0 时，grad 更准确地说是一个次梯度
    """
    y = recover_y_from_lambda(lambda_, c, beta, A, u, eps=eps)
    shifted_linear = c - A.T @ lambda_

    # 统一写成 shifted_linear * y - beta * y^2；beta=0 时自动退化为线性项
    phi = shifted_linear * y - beta * y * y
    g = float(np.dot(lambda_, R) + np.sum(phi))

    grad = R - A @ y
    return g, grad, y


def projected_gradient_mapping_norm(lambda_: np.ndarray, grad: np.ndarray) -> float:
    """
    投影梯度映射的无穷范数，用于判断收敛。

    说明：
    - 在全部 beta_i > 0 的平滑情形下，这是比较自然的最优性指标
    - 在存在 beta_i = 0 的混合情形下，它更适合被看作一个“站立性代理指标”
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

    eps = config.zero_beta_eps

    problem.validate_qp_compatible(eps=eps)
    x = validate_x_binary(x, problem.n)

    c = problem.c
    beta = problem.beta
    A = problem.consumption_matrix
    R = problem.resource_limit
    u = upper_bound_from_x(problem, x)

    has_zero_beta = np.any(beta <= eps)

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

    g, grad, y = dual_objective_and_gradient(lambda_, c, beta, A, R, u, eps=eps)

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
                    "mode": "subgradient" if has_zero_beta else "gradient",
                }
            )

        # -------------------------------------------
        # 情形 A：全部 beta_i > 0
        # 使用原来的投影梯度 + 回溯线搜索
        # -------------------------------------------
        if not has_zero_beta:
            if pg_norm <= config.tol and primal_violation <= 10.0 * config.tol:
                converged = True
                break

            step = config.step_init
            accepted = False

            while step >= config.min_step:
                candidate_lambda = np.maximum(lambda_ - step * grad, 0.0)
                candidate_g, candidate_grad, candidate_y = dual_objective_and_gradient(
                    candidate_lambda, c, beta, A, R, u, eps=eps
                )

                # 对偶目标下降则接受
                if candidate_g <= g + 1e-14:
                    accepted = True
                    break

                step *= config.backtrack_factor

            if not accepted:
                candidate_lambda = np.maximum(lambda_ - config.min_step * grad, 0.0)
                candidate_g, candidate_grad, candidate_y = dual_objective_and_gradient(
                    candidate_lambda, c, beta, A, R, u, eps=eps
                )

        # -------------------------------------------
        # 情形 B：存在 beta_i = 0
        # 使用投影次梯度法 + 衰减步长
        # -------------------------------------------
        else:
            # 次梯度法不再依赖平滑性和线搜索
            step = config.subgradient_step_scale / np.sqrt(it)

            candidate_lambda = np.maximum(lambda_ - step * grad, 0.0)
            candidate_g, candidate_grad, candidate_y = dual_objective_and_gradient(
                candidate_lambda, c, beta, A, R, u, eps=eps
            )

            # 混合情形下的停止条件更温和一些：
            # 既看资源违反，也看投影残差是否已经足够小
            if pg_norm <= max(config.tol, 1e-6) and primal_violation <= 10.0 * config.tol:
                converged = True
                break

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