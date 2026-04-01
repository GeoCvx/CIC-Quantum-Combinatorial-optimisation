# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class IsingModel:
    """
    Ising 模型表示：

        E(z) = z^T J z + h^T z + const

    其中：
        z ∈ {-1, +1}^n
        J: (n, n) 对称矩阵（通常只用上三角）
        h: (n,) 线性项
        const: 常数项（可忽略，但保留方便调试）
    """
    J: np.ndarray
    h: np.ndarray
    const: float


# =========================
# QUBO → Ising
# =========================

def qubo_to_ising(Q: np.ndarray) -> IsingModel:
    """
    将 QUBO：

        min x^T Q x,  x ∈ {0,1}^n

    转换为 Ising：

        min z^T J z + h^T z + const,  z ∈ {-1,+1}^n

    变量关系：
        x_i = (1 - z_i) / 2
    """

    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q 必须是方阵。")

    n = Q.shape[0]

    # 保证对称
    Q = 0.5 * (Q + Q.T)

    J = np.zeros_like(Q)
    h = np.zeros(n)
    const = 0.0

    # ===== 展开公式 =====
    #
    # x_i = (1 - z_i) / 2
    #
    # x^T Q x
    # = Σ_ij Q_ij * (1 - z_i)/2 * (1 - z_j)/2
    # = 1/4 Σ_ij Q_ij (1 - z_i - z_j + z_i z_j)
    #
    # 分解为：
    # 常数项 + 线性项 + 二次项

    for i in range(n):
        for j in range(n):
            q = Q[i, j]
            if abs(q) < 1e-15:
                continue

            # 常数项
            const += q * 0.25

            # 线性项
            h[i] += -q * 0.25
            h[j] += -q * 0.25

            # 二次项
            J[i, j] += q * 0.25

    # ===== 对角修正 =====
    # z_i^2 = 1 → 对角项应吸收到常数中
    for i in range(n):
        const += J[i, i]
        J[i, i] = 0.0

    # 再对称一次（数值稳定）
    J = 0.5 * (J + J.T)

    return IsingModel(J=J, h=h, const=const)


# =========================
# Ising → QUBO（可选，用于验证）
# =========================

def ising_to_qubo(model: IsingModel) -> np.ndarray:
    """
    将 Ising 模型转回 QUBO，用于验证正确性。
    """
    J = np.asarray(model.J, dtype=float)
    h = np.asarray(model.h, dtype=float)

    n = len(h)
    Q = np.zeros((n, n), dtype=float)

    # 反向代换：
    # z_i = 1 - 2 x_i

    for i in range(n):
        for j in range(n):
            Jij = J[i, j]
            if abs(Jij) < 1e-15:
                continue

            # z_i z_j = (1 - 2x_i)(1 - 2x_j)
            # = 1 - 2x_i - 2x_j + 4 x_i x_j

            Q[i, j] += 4.0 * Jij
            Q[i, i] += -2.0 * Jij
            Q[j, j] += -2.0 * Jij

    for i in range(n):
        hi = h[i]
        if abs(hi) < 1e-15:
            continue

        # z_i = 1 - 2x_i
        Q[i, i] += -2.0 * hi

    return 0.5 * (Q + Q.T)


# =========================
# 工具函数
# =========================

def qubo_to_ising_energy(Q: np.ndarray, x: np.ndarray) -> float:
    """
    QUBO 能量：
        x^T Q x
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    return float(x @ Q @ x)


def ising_energy(model: IsingModel, z: np.ndarray) -> float:
    """
    Ising 能量：
        z^T J z + h^T z + const
    """
    z = np.asarray(z, dtype=float).reshape(-1)
    return float(z @ model.J @ z + model.h @ z + model.const)


def x_to_z(x: np.ndarray) -> np.ndarray:
    """
    x ∈ {0,1} → z ∈ {-1,+1}
    """
    x = np.asarray(x, dtype=float)
    return 1.0 - 2.0 * x


def z_to_x(z: np.ndarray) -> np.ndarray:
    """
    z ∈ {-1,+1} → x ∈ {0,1}
    """
    z = np.asarray(z, dtype=float)
    return (1.0 - z) / 2.0


# =========================
# Quick self-test
# =========================

if __name__ == "__main__":
    import json
    import numpy as np
    from algorithms.quantum.qubo_builder import build_qubo

    # 读取一个测试问题（根据你的项目路径调整）
    with open("../../data/raw/problem_micp_1.json", "r", encoding="utf-8") as f:
        problem_dict = json.load(f)

    Q = build_qubo(problem_dict)
    model = qubo_to_ising(Q)

    # 随机测试
    rng = np.random.default_rng(42)
    x = rng.integers(0, 2, size=Q.shape[0]).astype(float)
    z = x_to_z(x)

    qubo_val = x @ Q @ x
    ising_val = ising_energy(model, z)

    print("QUBO:", qubo_val)
    print("Ising:", ising_val)
    print("Difference:", abs(qubo_val - ising_val))

    print("Ising:", ising_energy(model, z))