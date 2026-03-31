import numpy as np
from typing import List, Dict, Any

from algorithms.subproblem.router import evaluate_subproblem


# =========================
# 工具函数
# =========================

def random_binary_x(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=n).astype(float)


def greedy_initialization(problem_dict: dict) -> np.ndarray:
    """
    简单贪心：按 (p - alpha) / resource_usage 排序
    """
    price = np.array(problem_dict["price"])
    alpha = np.array(problem_dict["alpha"])
    A = np.array(problem_dict["consumption_matrix"])

    c = price - alpha

    # 避免除0
    resource_cost = np.sum(A, axis=0) + 1e-8
    score = c / resource_cost

    order = np.argsort(-score)
    x = np.zeros(len(c))

    best_z = -1e18

    for i in order:
        x[i] = 1
        try:
            result = evaluate_subproblem(problem_dict, x)
            if result["Z"] > best_z:
                best_z = result["Z"]
            else:
                x[i] = 0  # rollback
        except Exception:
            x[i] = 0

    return x


# =========================
# 局部搜索（Hill Climbing）
# =========================

def local_search(
    problem_dict: dict,
    x_init: np.ndarray,
    max_iter: int = 100,
) -> Dict[str, Any]:
    x = x_init.copy()

    best_result = evaluate_subproblem(problem_dict, x)
    best_z = best_result["Z"]

    n = len(x)

    for _ in range(max_iter):
        improved = False

        for i in range(n):
            x_new = x.copy()
            x_new[i] = 1 - x_new[i]  # flip

            try:
                result = evaluate_subproblem(problem_dict, x_new)
                z = result["Z"]
            except Exception:
                continue

            if z > best_z:
                x = x_new
                best_z = z
                best_result = result
                improved = True
                break

        if not improved:
            break

    return {
        "best_x": x,
        "best_result": best_result,
        "best_objective": best_z,
    }


# =========================
# Elite Pool
# =========================

class ElitePool:
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.pool: List[Dict[str, Any]] = []

    def add(self, candidate: Dict[str, Any]):
        self.pool.append(candidate)
        self.pool = sorted(self.pool, key=lambda x: -x["best_objective"])
        if len(self.pool) > self.max_size:
            self.pool.pop()

    def get_best(self):
        return self.pool[0] if self.pool else None


# =========================
# 主接口（替代 random search）
# =========================

def search_x_classical(
    problem_dict: dict,
    num_starts: int = 10,
    local_iter: int = 100,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    n = problem_dict["product_count"]

    elite = ElitePool(max_size=5)

    # ===== 1. 贪心初始化 =====
    try:
        x0 = greedy_initialization(problem_dict)
        result0 = local_search(problem_dict, x0, max_iter=local_iter)
        elite.add(result0)
    except Exception:
        pass

    # ===== 2. 多起点 =====
    for _ in range(num_starts):
        x_rand = random_binary_x(n, rng)

        try:
            result = local_search(problem_dict, x_rand, max_iter=local_iter)
            elite.add(result)
        except Exception:
            continue

    best = elite.get_best()

    return {
        "best_x": best["best_x"],
        "best_result": best["best_result"],
        "best_objective": best["best_objective"],
        "elite_pool": elite.pool,
    }
