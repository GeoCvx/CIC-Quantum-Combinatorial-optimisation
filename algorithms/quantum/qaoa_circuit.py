# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from pyqpanda3.core import QCircuit, QProg, H, RX, RZ, CNOT, measure

from algorithms.quantum.ising_mapping import IsingModel


@dataclass
class QAOACircuitSpec:
    """
    QAOA 电路规格。
    """
    p: int
    gammas: np.ndarray
    betas: np.ndarray

    def validate(self) -> None:
        if self.p <= 0:
            raise ValueError("p 必须为正整数。")
        if self.gammas.shape != (self.p,):
            raise ValueError(f"gammas shape 应为 ({self.p},)，实际为 {self.gammas.shape}")
        if self.betas.shape != (self.p,):
            raise ValueError(f"betas shape 应为 ({self.p},)，实际为 {self.betas.shape}")


def _as_1d_float_array(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} 必须是一维数组。")
    return arr


def _validate_ising_model(model: IsingModel) -> None:
    J = np.asarray(model.J, dtype=float)
    h = np.asarray(model.h, dtype=float).reshape(-1)

    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError("IsingModel.J 必须是方阵。")
    if h.shape != (J.shape[0],):
        raise ValueError(f"IsingModel.h shape 应为 ({J.shape[0]},)，实际为 {h.shape}")


def make_qaoa_spec(
    p: int,
    gammas: Sequence[float] | None = None,
    betas: Sequence[float] | None = None,
) -> QAOACircuitSpec:
    """
    生成 QAOA 参数规格。
    若未提供 gammas / betas，则用简单默认值初始化。
    """
    if p <= 0:
        raise ValueError("p 必须为正整数。")

    if gammas is None:
        gammas_arr = np.full(p, 0.5, dtype=float)
    else:
        gammas_arr = _as_1d_float_array(gammas, "gammas")
        if gammas_arr.shape != (p,):
            raise ValueError(f"gammas shape 应为 ({p},)，实际为 {gammas_arr.shape}")

    if betas is None:
        betas_arr = np.full(p, 0.5, dtype=float)
    else:
        betas_arr = _as_1d_float_array(betas, "betas")
        if betas_arr.shape != (p,):
            raise ValueError(f"betas shape 应为 ({p},)，实际为 {betas_arr.shape}")

    spec = QAOACircuitSpec(
        p=p,
        gammas=gammas_arr,
        betas=betas_arr,
    )
    spec.validate()
    return spec


def build_initial_layer(n_qubits: int) -> QCircuit:
    """
    初始层：对所有量子比特施加 Hadamard，得到均匀叠加态。
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits 必须为正整数。")

    circuit = QCircuit()
    for q in range(n_qubits):
        circuit << H(q)
    return circuit


def apply_cost_layer(
    circuit: QCircuit,
    model: IsingModel,
    gamma: float,
) -> QCircuit:
    """
    Cost Hamiltonian 层，对应 Ising 能量：

        E(z) = z^T J z + h^T z + const

    这里只实现：
    - 单体项 h_i Z_i
    - 双体项 J_ij Z_i Z_j

    采用实现：
    - h_i Z_i       -> RZ(i, 2 * gamma * h_i)
    - J_ij Z_i Z_j  -> CNOT(i, j) -> RZ(j, 2 * gamma * J_ij) -> CNOT(i, j)

    如果你后面发现优化方向反了，通常只需要整体把 gamma 改号，
    或把这里的角度符号整体翻转即可。
    """
    _validate_ising_model(model)

    J = np.asarray(model.J, dtype=float)
    h = np.asarray(model.h, dtype=float).reshape(-1)
    n = len(h)

    # 单体项
    for i in range(n):
        coeff = float(h[i])
        if abs(coeff) > 1e-15:
            circuit << RZ(i, 2.0 * gamma * coeff)

    # 双体项（只扫描上三角）
    for i in range(n):
        for j in range(i + 1, n):
            coeff = float(J[i, j])
            if abs(coeff) <= 1e-15:
                continue
            circuit << CNOT(i, j)
            circuit << RZ(j, 2.0 * gamma * coeff)
            circuit << CNOT(i, j)

    return circuit


def apply_mixer_layer(
    circuit: QCircuit,
    n_qubits: int,
    beta: float,
) -> QCircuit:
    """
    Mixer Hamiltonian 层：
        H_M = sum_i X_i

    常见实现：
        RX(i, 2 * beta)
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits 必须为正整数。")

    for q in range(n_qubits):
        circuit << RX(q, 2.0 * beta)

    return circuit


def build_qaoa_circuit(
    model: IsingModel,
    p: int,
    gammas: Sequence[float] | None = None,
    betas: Sequence[float] | None = None,
) -> QCircuit:
    """
    构造不带测量的 QAOA 主电路。
    """
    _validate_ising_model(model)
    n = len(model.h)
    spec = make_qaoa_spec(p=p, gammas=gammas, betas=betas)

    circuit = QCircuit()
    circuit << build_initial_layer(n)

    for layer in range(spec.p):
        gamma = float(spec.gammas[layer])
        beta = float(spec.betas[layer])

        cost_layer = QCircuit()
        apply_cost_layer(cost_layer, model, gamma)
        circuit << cost_layer

        mixer_layer = QCircuit()
        apply_mixer_layer(mixer_layer, n, beta)
        circuit << mixer_layer

    return circuit


def build_qaoa_program(
    model: IsingModel,
    p: int,
    gammas: Sequence[float] | None = None,
    betas: Sequence[float] | None = None,
    add_measure: bool = True,
) -> QProg:
    """
    构造完整 QAOA 程序。
    若 add_measure=True，则在末尾添加逐比特测量。
    """
    _validate_ising_model(model)
    n = len(model.h)

    prog = QProg()
    circuit = build_qaoa_circuit(
        model=model,
        p=p,
        gammas=gammas,
        betas=betas,
    )
    prog << circuit

    if add_measure:
        for q in range(n):
            prog << measure(q, q)

    return prog


def default_parameter_grid(
    p: int,
    gamma_value: float = 0.5,
    beta_value: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成一组默认参数。
    """
    if p <= 0:
        raise ValueError("p 必须为正整数。")
    gammas = np.full(p, gamma_value, dtype=float)
    betas = np.full(p, beta_value, dtype=float)
    return gammas, betas


if __name__ == "__main__":
    # 本地快速自检
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

    gammas, betas = default_parameter_grid(p=2, gamma_value=0.6, beta_value=0.4)

    circuit = build_qaoa_circuit(model, p=2, gammas=gammas, betas=betas)
    prog = build_qaoa_program(model, p=2, gammas=gammas, betas=betas, add_measure=True)

    print("=== QCircuit ===")
    print(circuit)
    print("=== QProg ===")
    print(prog)