# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CandidateSelectionConfig:
    top_k: int = 8
    min_hamming_distance: int = 2
    prefer_higher_probability: bool = True

    # 新增：与 incumbent 的距离控制
    min_far_hamming_distance: int = 3
    min_far_candidates: int = 2

    # 新增：tabu 近邻过滤
    tabu_hamming_threshold: int = 1

    # 新增：候选综合排序权重
    objective_bonus: float = 0.05
    far_bonus: float = 0.02


def _hamming_distance(x1: np.ndarray, x2: np.ndarray) -> int:
    a = np.asarray(x1, dtype=float).reshape(-1)
    b = np.asarray(x2, dtype=float).reshape(-1)
    return int(np.sum(np.abs(a - b) > 0.5))


def _bitstring_from_x(x: np.ndarray) -> str:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return "".join(str(int(round(v))) for v in arr)


def _is_near_tabu(
    x: np.ndarray,
    tabu_set: list[str] | None,
    threshold: int,
) -> bool:
    if not tabu_set:
        return False
    x_bits = _bitstring_from_x(x)
    for key in tabu_set:
        n = min(len(key), len(x_bits))
        dist = sum(1 for i in range(n) if key[i] != x_bits[i])
        if dist <= threshold:
            return True
    return False


def _score_candidate(
    item: dict,
    incumbent: np.ndarray | None,
    config: CandidateSelectionConfig,
) -> float:
    """
    分数越小越优：
    - 低 energy 优先
    - probability 高优先
    - objective_value 高优先（若已有真实评估）
    - 与 incumbent 距离远一点的候选，得到轻微加成
    """
    raw_energy = item.get("energy", None)
    energy = float(raw_energy) if raw_energy is not None else float("inf")

    probability = item.get("probability", None)
    prob_term = (
        -float(probability)
        if (config.prefer_higher_probability and probability is not None)
        else 0.0
    )

    objective_value = item.get("objective_value", None)
    obj_term = (
        -config.objective_bonus * float(objective_value)
        if objective_value is not None
        else 0.0
    )

    far_term = 0.0
    if incumbent is not None:
        dist = _hamming_distance(item["x"], incumbent)
        far_term = -config.far_bonus * float(dist)

    return energy + prob_term + obj_term + far_term


def _append_if_diverse(
    selected: list[dict],
    item: dict,
    min_hamming_distance: int,
) -> bool:
    x = np.asarray(item["x"], dtype=float).reshape(-1)
    for kept in selected:
        if _hamming_distance(x, kept["x"]) < min_hamming_distance:
            return False

    new_item = dict(item)
    new_item["x"] = x.copy()
    selected.append(new_item)
    return True


def select_candidates(
    candidates: list[dict],
    config: CandidateSelectionConfig | None = None,
    incumbent: np.ndarray | None = None,
    tabu_set: list[str] | None = None,
) -> list[dict]:
    if config is None:
        config = CandidateSelectionConfig()

    if not candidates:
        return []

    ranked = sorted(
        candidates,
        key=lambda item: _score_candidate(item, incumbent=incumbent, config=config),
    )

    # 先去重 bitstring
    deduped: list[dict] = []
    seen = set()
    for item in ranked:
        x = np.asarray(item["x"], dtype=float).reshape(-1)
        bitstring = item.get("bitstring") or _bitstring_from_x(x)
        if bitstring in seen:
            continue
        seen.add(bitstring)

        if _is_near_tabu(x, tabu_set, config.tabu_hamming_threshold):
            # tabu 近邻直接跳过
            continue

        new_item = dict(item)
        new_item["x"] = x.copy()
        new_item["bitstring"] = bitstring
        deduped.append(new_item)

    if not deduped:
        # 极端情况下保底返回一个原候选
        first = dict(candidates[0])
        first["x"] = np.asarray(first["x"], dtype=float).copy()
        first["bitstring"] = first.get("bitstring") or _bitstring_from_x(first["x"])
        return [first]

    selected: list[dict] = []

    # ===== 第一阶段：强制保留远离 incumbent 的候选 =====
    if incumbent is not None:
        far_candidates = []
        for item in deduped:
            dist = _hamming_distance(item["x"], incumbent)
            if dist >= config.min_far_hamming_distance:
                far_candidates.append((dist, item))

        far_candidates.sort(key=lambda pair: (-pair[0], _score_candidate(pair[1], incumbent, config)))

        for _, item in far_candidates:
            if len(selected) >= min(config.min_far_candidates, config.top_k):
                break
            _append_if_diverse(selected, item, config.min_hamming_distance)

    # ===== 第二阶段：常规按综合排序补齐 =====
    for item in deduped:
        if len(selected) >= config.top_k:
            break
        _append_if_diverse(selected, item, config.min_hamming_distance)

    # ===== 兜底 =====
    if not selected:
        item = dict(deduped[0])
        item["x"] = np.asarray(item["x"], dtype=float).copy()
        selected = [item]

    return selected