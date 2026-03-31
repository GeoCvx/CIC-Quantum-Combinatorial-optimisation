# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import List, Dict, Any, Sequence
import numpy as np

from algorithms.quantum.ising_mapping import IsingModel
from algorithms.quantum.qaoa_runner import (
    QAOARunConfig,
    QAOARunResult,
    run_qaoa_once,
    top_k_samples,
)


@dataclass
class QAOASolverConfig:
    """
    QAOA 参数搜索配置。
    """
    p: int = 1
    shots: int = 512
    seed: int | None = 42

    # 网格搜索参数
    gamma_values: np.ndarray | None = None
    beta_values: np.ndarray | None = None

    # 若未显式给出 gamma_values / beta_values，则使用 linspace 生成
    gamma_min: float = 0.1
    gamma_max: float = 1.0
    gamma_points: int = 5

    beta_min: float = 0.1
    beta_max: float = 1.0
    beta_points: int = 5

    # 每次 run 保留多少样本供全局聚合
    per_run_top_k: int = 10

    # 全局最终保留多少样本
    final_top_k: int = 20


@dataclass
class QAOASolverResult:
    """
    QAOA 多参数搜索结果。
    """
    best_x: np.ndarray
    best_bitstring: str
    best_energy: float

    best_gammas: np.ndarray
    best_betas: np.ndarray

    best_run: QAOARunResult
    all_runs: List[dict] = field(default_factory=list)
    top_samples: List[dict] = field(default_factory=list)


def _validate_solver_config(config: QAOASolverConfig) -> None:
    if config.p <= 0:
        raise ValueError("p 必须为正整数。")
    if config.shots <= 0:
        raise ValueError("shots 必须为正整数。")
    if config.per_run_top_k <= 0:
        raise ValueError("per_run_top_k 必须为正整数。")
    if config.final_top_k <= 0:
        raise ValueError("final_top_k 必须为正整数。")

    if config.gamma_values is None and config.gamma_points <= 0:
        raise ValueError("gamma_points 必须为正整数。")
    if config.beta_values is None and config.beta_points <= 0:
        raise ValueError("beta_points 必须为正整数。")


def _make_1d_grid(
    values: Sequence[float] | None,
    vmin: float,
    vmax: float,
    points: int,
    name: str,
) -> np.ndarray:
    if values is not None:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"{name} 不能为空。")
        return arr

    if points <= 0:
        raise ValueError(f"{name} 的 points 必须为正整数。")

    return np.linspace(vmin, vmax, points, dtype=float)


def _expand_parameter_grid_for_p(
    base_values: np.ndarray,
    p: int,
) -> List[np.ndarray]:
    """
    将一维基础候选值扩展成长度为 p 的参数向量列表。

    当前实现采用“各层同参”的简化策略：
        gamma_vec = [g, g, ..., g]
        beta_vec  = [b, b, ..., b]

    这样搜索空间较小，更适合比赛初版。
    """
    result = []
    for v in base_values:
        result.append(np.full(p, float(v), dtype=float))
    return result


def _deduplicate_samples(samples: List[dict]) -> List[dict]:
    """
    按 bitstring 去重，保留能量更低的；若能量相同则保留概率更高的。
    """
    best_by_bitstring: Dict[str, dict] = {}

    for sample in samples:
        key = sample["bitstring"]
        if key not in best_by_bitstring:
            best_by_bitstring[key] = sample
            continue

        old = best_by_bitstring[key]
        if sample["energy"] < old["energy"]:
            best_by_bitstring[key] = sample
        elif sample["energy"] == old["energy"] and sample["probability"] > old["probability"]:
            best_by_bitstring[key] = sample

    deduped = list(best_by_bitstring.values())
    deduped.sort(key=lambda s: (s["energy"], -s["probability"]))
    return deduped


def solve_qaoa(
    model: IsingModel,
    config: QAOASolverConfig | None = None,
) -> QAOASolverResult:
    """
    对给定 Ising 模型做 QAOA 参数网格搜索，返回全局最优候选。
    """
    if config is None:
        config = QAOASolverConfig()

    _validate_solver_config(config)

    gamma_base = _make_1d_grid(
        values=config.gamma_values,
        vmin=config.gamma_min,
        vmax=config.gamma_max,
        points=config.gamma_points,
        name="gamma_values",
    )
    beta_base = _make_1d_grid(
        values=config.beta_values,
        vmin=config.beta_min,
        vmax=config.beta_max,
        points=config.beta_points,
        name="beta_values",
    )

    gamma_grid = _expand_parameter_grid_for_p(gamma_base, config.p)
    beta_grid = _expand_parameter_grid_for_p(beta_base, config.p)

    all_runs: List[dict] = []
    all_top_samples: List[dict] = []

    global_best_energy = float("inf")
    global_best_x: np.ndarray | None = None
    global_best_bitstring: str | None = None
    global_best_gammas: np.ndarray | None = None
    global_best_betas: np.ndarray | None = None
    global_best_run: QAOARunResult | None = None

    for gammas, betas in product(gamma_grid, beta_grid):
        run_config = QAOARunConfig(
            shots=config.shots,
            p=config.p,
            gammas=gammas,
            betas=betas,
            seed=config.seed,
        )

        run_result = run_qaoa_once(model, config=run_config)

        run_summary = {
            "gammas": gammas.copy(),
            "betas": betas.copy(),
            "best_bitstring": run_result.best_bitstring,
            "best_x": run_result.best_x.copy(),
            "best_energy": float(run_result.best_energy),
            "num_unique_samples": len(run_result.samples),
            "run_result": run_result,
        }
        all_runs.append(run_summary)

        # 聚合每次 run 的 top-k 样本，并附上参数信息
        for sample in top_k_samples(run_result, k=config.per_run_top_k):
            enriched = dict(sample)
            enriched["gammas"] = gammas.copy()
            enriched["betas"] = betas.copy()
            all_top_samples.append(enriched)

        if run_result.best_energy < global_best_energy:
            global_best_energy = float(run_result.best_energy)
            global_best_x = run_result.best_x.copy()
            global_best_bitstring = run_result.best_bitstring
            global_best_gammas = gammas.copy()
            global_best_betas = betas.copy()
            global_best_run = run_result

    if global_best_x is None or global_best_bitstring is None:
        raise RuntimeError("QAOA 参数搜索未产生任何有效结果。")

    deduped_top_samples = _deduplicate_samples(all_top_samples)
    final_top_samples = deduped_top_samples[: config.final_top_k]

    # all_runs 也按 best_energy 排序，方便查看
    all_runs.sort(key=lambda r: r["best_energy"])

    return QAOASolverResult(
        best_x=global_best_x,
        best_bitstring=global_best_bitstring,
        best_energy=global_best_energy,
        best_gammas=global_best_gammas,
        best_betas=global_best_betas,
        best_run=global_best_run,
        all_runs=all_runs,
        top_samples=final_top_samples,
    )


if __name__ == "__main__":
    # 本地自检
    J = np.array(
        [
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.5],
            [0.0, 0.5, 0.0],
        ],
        dtype=float,
    )
    h = np.array([0.2, -0.1, 0.3], dtype=float)
    model = IsingModel(J=J, h=h, const=0.0)

    config = QAOASolverConfig(
        p=1,
        shots=256,
        gamma_values=np.array([0.2, 0.5, 0.8], dtype=float),
        beta_values=np.array([0.2, 0.5, 0.8], dtype=float),
        seed=42,
        per_run_top_k=5,
        final_top_k=10,
    )

    result = solve_qaoa(model, config=config)

    print("=== Global Best ===")
    print("best_bitstring:", result.best_bitstring)
    print("best_x:", result.best_x.tolist())
    print("best_energy:", result.best_energy)
    print("best_gammas:", result.best_gammas.tolist())
    print("best_betas:", result.best_betas.tolist())

    print("\n=== Best Runs ===")
    for run in result.all_runs[:5]:
        print(
            {
                "best_energy": round(run["best_energy"], 6),
                "best_bitstring": run["best_bitstring"],
                "gammas": run["gammas"].tolist(),
                "betas": run["betas"].tolist(),
                "num_unique_samples": run["num_unique_samples"],
            }
        )

    print("\n=== Top Samples ===")
    for sample in result.top_samples[:10]:
        print(
            {
                "bitstring": sample["bitstring"],
                "probability": round(sample["probability"], 4),
                "energy": round(sample["energy"], 6),
                "gammas": sample["gammas"].tolist(),
                "betas": sample["betas"].tolist(),
            }
        )