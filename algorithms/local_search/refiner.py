# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from algorithms.subproblem.router import evaluate_subproblem


@dataclass
class LocalRefineConfig:
    max_iter: int = 20
    enable_swap: bool = True

    # 新增：多起点精修
    max_start_points: int = 3

    # 新增：可传入 bias，引导邻域顺序
    use_guided_order: bool = True


def _safe_eval(problem_dict: dict, x: np.ndarray, round_digits: int = 6) -> dict[str, Any] | None:
    try:
        result = evaluate_subproblem(problem_dict, x, round_digits=round_digits)
        return {
            "x": np.asarray(x, dtype=float).copy(),
            "best_result": result,
            "objective_value": float(result["Z"]),
            "feasible": True,
        }
    except Exception:
        return None


def _ordered_flip_indices(x: np.ndarray, linear_bias: np.ndarray | None = None) -> list[int]:
    n = len(x)
    if linear_bias is None:
        return list(range(n))

    x = np.asarray(x, dtype=float).reshape(-1)
    linear_bias = np.asarray(linear_bias, dtype=float).reshape(-1)

    # 对未选中变量，优先尝试 bias 高的引入
    inactive = [i for i in range(n) if x[i] < 0.5]
    inactive.sort(key=lambda i: linear_bias[i], reverse=True)

    # 对已选中变量，优先尝试 bias 低的移除
    active = [i for i in range(n) if x[i] > 0.5]
    active.sort(key=lambda i: linear_bias[i])

    return inactive + active


def _refine_single_start(
    problem_dict: dict,
    x_init: np.ndarray,
    config: LocalRefineConfig,
    linear_bias: np.ndarray | None = None,
) -> dict[str, Any]:
    x = np.asarray(x_init, dtype=float).copy()
    current = _safe_eval(problem_dict, x)
    if current is None:
        raise RuntimeError("局部精修的初始解不可评估。")

    n = len(x)

    for _ in range(config.max_iter):
        improved = False

        # ===== 单点翻转 =====
        flip_order = (
            _ordered_flip_indices(x, linear_bias=linear_bias)
            if config.use_guided_order
            else list(range(n))
        )

        for i in flip_order:
            x_new = x.copy()
            x_new[i] = 1.0 - x_new[i]
            rec = _safe_eval(problem_dict, x_new)
            if rec is not None and rec["objective_value"] > current["objective_value"]:
                x = x_new
                current = rec
                improved = True
                break

        if improved:
            continue

        # ===== 双点交换 =====
        if config.enable_swap:
            active = np.where(x > 0.5)[0].tolist()
            inactive = np.where(x < 0.5)[0].tolist()

            if config.use_guided_order and linear_bias is not None:
                active.sort(key=lambda i: linear_bias[i])           # 先尝试移除低 bias 的已选变量
                inactive.sort(key=lambda i: linear_bias[i], reverse=True)  # 先尝试加入高 bias 的未选变量

            for i in active:
                for j in inactive:
                    x_new = x.copy()
                    x_new[i] = 0.0
                    x_new[j] = 1.0
                    rec = _safe_eval(problem_dict, x_new)
                    if rec is not None and rec["objective_value"] > current["objective_value"]:
                        x = x_new
                        current = rec
                        improved = True
                        break
                if improved:
                    break

        if not improved:
            break

    return current


def refine_solution_locally(
    problem_dict: dict,
    x_init: np.ndarray | list[np.ndarray],
    config: LocalRefineConfig | None = None,
    linear_bias: np.ndarray | None = None,
) -> dict[str, Any]:
    if config is None:
        config = LocalRefineConfig()

    # 允许单起点或多起点
    if isinstance(x_init, list):
        start_points = [np.asarray(x, dtype=float).copy() for x in x_init[: config.max_start_points]]
    else:
        start_points = [np.asarray(x_init, dtype=float).copy()]

    best = None
    best_obj = -float("inf")

    for x0 in start_points:
        rec = _refine_single_start(
            problem_dict,
            x0,
            config=config,
            linear_bias=linear_bias,
        )
        if rec["objective_value"] > best_obj:
            best_obj = rec["objective_value"]
            best = rec

    if best is None:
        raise RuntimeError("局部精修失败：所有起点均不可评估。")

    return best