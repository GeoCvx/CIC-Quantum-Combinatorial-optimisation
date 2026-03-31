# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np

from pyqpanda3.core import CPUQVM

from algorithms.quantum.ising_mapping import (
    IsingModel,
    ising_energy,
)
from algorithms.quantum.qaoa_circuit import build_qaoa_program


@dataclass
class QAOARunConfig:
    """
    QAOA 单次运行配置。
    """
    shots: int = 512
    p: int = 1
    gammas: np.ndarray | None = None
    betas: np.ndarray | None = None
    seed: int | None = None


@dataclass
class QAOARunResult:
    """
    单次 QAOA 运行结果。
    """
    counts: Dict[str, int]
    samples: List[dict]
    best_bitstring: str
    best_x: np.ndarray
    best_energy: float


def _validate_run_config(config: QAOARunConfig) -> None:
    if config.shots <= 0:
        raise ValueError("shots 必须为正整数。")
    if config.p <= 0:
        raise ValueError("p 必须为正整数。")


def _bitstring_to_z(bitstring: str) -> np.ndarray:
    """
    将测量比特串转换为 z ∈ {-1,+1}。

    约定：
        bit '0' -> z = +1
        bit '1' -> z = -1

    这样与 x = (1 - z) / 2 保持一致：
        0 -> x=0
        1 -> x=1
    """
    return np.array([1.0 if b == "0" else -1.0 for b in bitstring], dtype=float)


def _bitstring_to_x(bitstring: str) -> np.ndarray:
    """
    将测量比特串直接转成 x ∈ {0,1}^n。
    """
    return np.array([float(int(b)) for b in bitstring], dtype=float)


def _normalize_counts(raw_counts: Any, n_qubits: int) -> Dict[str, int]:
    """
    尝试把 pyqpanda3 返回的计数字典标准化为：
        {bitstring: count}

    兼容一些可能的 key 形式差异。
    """
    if raw_counts is None:
        raise RuntimeError("量子机未返回测量结果。")

    counts: Dict[str, int] = {}

    if isinstance(raw_counts, dict):
        for k, v in raw_counts.items():
            key = str(k).replace(" ", "")
            if len(key) != n_qubits:
                continue
            counts[key] = int(v)
    else:
        raise TypeError(f"不支持的 counts 类型: {type(raw_counts)}")

    if not counts:
        raise RuntimeError(
            "测量结果为空，可能是 pyqpanda3 返回格式与当前兼容逻辑不一致。"
        )

    return counts


def _extract_counts_from_qvm(qvm: CPUQVM, prog, shots: int) -> Dict[str, int]:
    """
    执行程序并提取测量结果。

    优先采用 pyqpanda3 官方示例风格：
        qvm.run(prog, shots)
        qvm.result().get_counts()
    """
    if hasattr(qvm, "run"):
        try:
            qvm.run(prog, shots)
        except TypeError:
            # 某些实现也许只支持 run(prog)
            qvm.run(prog)

        if hasattr(qvm, "result"):
            result = qvm.result()
            if hasattr(result, "get_counts"):
                return result.get_counts()

        if hasattr(qvm, "get_result"):
            result = qvm.get_result()
            if hasattr(result, "get_counts"):
                return result.get_counts()

    if hasattr(qvm, "run_with_configuration"):
        raw_counts = qvm.run_with_configuration(prog, shots)
        return raw_counts

    raise RuntimeError("当前 pyqpanda3 版本下未找到可用的测量执行接口。")


def run_qaoa_once(
    model: IsingModel,
    config: QAOARunConfig | None = None,
) -> QAOARunResult:
    """
    单次运行 QAOA 并返回采样结果。
    """
    if config is None:
        config = QAOARunConfig()

    _validate_run_config(config)

    n = len(model.h)

    qvm = CPUQVM()

    if config.seed is not None and hasattr(qvm, "set_random_seed"):
        qvm.set_random_seed(config.seed)

    prog = build_qaoa_program(
        model=model,
        p=config.p,
        gammas=config.gammas,
        betas=config.betas,
        add_measure=True,
    )

    raw_counts = _extract_counts_from_qvm(qvm, prog, config.shots)
    counts = _normalize_counts(raw_counts, n_qubits=n)

    samples: List[dict] = []
    best_energy = float("inf")
    best_bitstring: str | None = None
    best_x: np.ndarray | None = None

    for bitstring, count in counts.items():
        x = _bitstring_to_x(bitstring)
        z = _bitstring_to_z(bitstring)
        energy = ising_energy(model, z)

        sample = {
            "bitstring": bitstring,
            "count": int(count),
            "probability": float(count / config.shots),
            "x": x,
            "z": z,
            "energy": float(energy),
        }
        samples.append(sample)

        if energy < best_energy:
            best_energy = float(energy)
            best_bitstring = bitstring
            best_x = x.copy()

    # 按能量升序，再按概率降序排序
    samples.sort(key=lambda s: (s["energy"], -s["probability"]))

    if hasattr(qvm, "finalize"):
        try:
            qvm.finalize()
        except Exception:
            pass

    if best_bitstring is None or best_x is None:
        raise RuntimeError("QAOA 未产生任何有效样本。")

    return QAOARunResult(
        counts=counts,
        samples=samples,
        best_bitstring=best_bitstring,
        best_x=best_x,
        best_energy=best_energy,
    )


def top_k_samples(
    result: QAOARunResult,
    k: int = 10,
) -> List[dict]:
    """
    返回前 k 个样本。
    """
    if k <= 0:
        raise ValueError("k 必须为正整数。")
    return result.samples[:k]

def brute_force_ising(model: IsingModel):
    n = len(model.h)
    all_samples = []

    for i in range(2 ** n):
        bitstring = format(i, f"0{n}b")
        z = np.array([1.0 if b == "0" else -1.0 for b in bitstring], dtype=float)
        x = np.array([float(int(b)) for b in bitstring], dtype=float)
        energy = ising_energy(model, z)
        all_samples.append((bitstring, x, energy))

    all_samples.sort(key=lambda t: t[2])
    return all_samples

if __name__ == "__main__":
    # 本地自检：构造一个小 Ising 模型并运行一次
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

    config = QAOARunConfig(
        shots=256,
        p=1,
        gammas=np.array([0.6], dtype=float),
        betas=np.array([0.4], dtype=float),
        seed=42,
    )

    result = run_qaoa_once(model, config=config)

    print("=== Best Sample ===")
    print("bitstring:", result.best_bitstring)
    print("x:", result.best_x.tolist())
    print("energy:", result.best_energy)

    print("\n=== Top Samples ===")
    for sample in top_k_samples(result, k=5):
        print(
            {
                "bitstring": sample["bitstring"],
                "count": sample["count"],
                "probability": round(sample["probability"], 4),
                "energy": round(sample["energy"], 6),
            }
        )

    print("\n=== Brute Force Check ===")
    for bitstring, x, energy in brute_force_ising(model):
        print(bitstring, x.tolist(), energy)