# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ProblemData:
    """
    固定 x 后 LP 子问题所需的数据。

    字段与比赛 JSON 一一对应：
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
        固定 x 后 LP 的线性利润系数：
            c = p - alpha
        """
        return self.price - self.alpha

    @classmethod
    def from_dict(cls, data: dict) -> "ProblemData":
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

    def validate_lp_compatible(self, eps: float = 1e-12) -> None:
        """
        当前 LP 子问题要求 beta 全为 0。
        """
        if np.any(np.abs(self.beta) > eps):
            bad_idx = np.where(np.abs(self.beta) > eps)[0].tolist()
            raise ValueError(
                "当前 LP 子问题仅支持 beta_i = 0。"
                f" 以下位置 |beta_i| > {eps}: {bad_idx}"
            )