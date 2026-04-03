# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ProblemData:
    """
    固定 x 后 QP 子问题所需的全部问题数据。

    字段与比赛 JSON 对应：
    - product_count
    - resource_count
    - price
    - fixed_cost
    - alpha
    - beta
    - max_demand
    - resource_limit
    - consumption_matrix
    """
    product_count: int
    resource_count: int
    price: np.ndarray
    fixed_cost: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    max_demand: np.ndarray
    resource_limit: np.ndarray
    consumption_matrix: np.ndarray

    @property
    def n(self) -> int:
        return self.product_count

    @property
    def m(self) -> int:
        return self.resource_count

    @property
    def c(self) -> np.ndarray:
        """
        固定 x 后 QP 的线性系数：
            c = p - alpha
        """
        return self.price - self.alpha

    @classmethod
    def from_dict(cls, data: dict) -> "ProblemData":
        """
        从 json.load(...) 得到的字典构造 ProblemData。
        """
        obj = cls(
            product_count=int(data["product_count"]),
            resource_count=int(data["resource_count"]),
            price=np.asarray(data["price"], dtype=float),
            fixed_cost=np.asarray(data["fixed_cost"], dtype=float),
            alpha=np.asarray(data["alpha"], dtype=float),
            beta=np.asarray(data["beta"], dtype=float),
            max_demand=np.asarray(data["max_demand"], dtype=float),
            resource_limit=np.asarray(data["resource_limit"], dtype=float),
            consumption_matrix=np.asarray(data["consumption_matrix"], dtype=float),
        )
        obj.validate_basic()
        return obj

    def validate_basic(self) -> None:
        """
        基本维度检查。
        """
        n = self.product_count
        m = self.resource_count

        if self.price.shape != (n,):
            raise ValueError(f"price shape 应为 ({n},)，实际为 {self.price.shape}")
        if self.fixed_cost.shape != (n,):
            raise ValueError(f"fixed_cost shape 应为 ({n},)，实际为 {self.fixed_cost.shape}")
        if self.alpha.shape != (n,):
            raise ValueError(f"alpha shape 应为 ({n},)，实际为 {self.alpha.shape}")
        if self.beta.shape != (n,):
            raise ValueError(f"beta shape 应为 ({n},)，实际为 {self.beta.shape}")
        if self.max_demand.shape != (n,):
            raise ValueError(f"max_demand shape 应为 ({n},)，实际为 {self.max_demand.shape}")
        if self.resource_limit.shape != (m,):
            raise ValueError(f"resource_limit shape 应为 ({m},)，实际为 {self.resource_limit.shape}")
        if self.consumption_matrix.shape != (m, n):
            raise ValueError(
                f"consumption_matrix shape 应为 ({m}, {n})，实际为 {self.consumption_matrix.shape}"
            )

        if np.any(self.max_demand < 0):
            raise ValueError("max_demand 中不应出现负数。")
        if np.any(self.resource_limit < 0):
            raise ValueError("resource_limit 中不应出现负数。")

    def validate_qp_compatible(self, eps: float = 1e-12) -> None:
        """
        当前 QP 求解器支持 beta 不全为 0 的情形。
        若 beta 全为 0，则该实例应交给 LP 求解器处理。

        注意：
        - 这里不再禁止存在部分 beta_i = 0
        - 混合情形（部分为 0，部分 > 0）由 dual_solver 内部处理
        """
        if np.all(self.beta <= eps):
            raise ValueError(
                "当前实例中 beta 全为 0，应交给 LP 求解器处理，而不是 QP 求解器。"
            )