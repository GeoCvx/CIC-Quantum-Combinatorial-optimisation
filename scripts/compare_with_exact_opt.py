# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import json
import time
import argparse
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np


# =========================================================
# 把项目根目录加入 PYTHONPATH
# 兼容两种放置方式：
# 1) 放在项目根目录下
# 2) 放在 scripts/ 目录下
# =========================================================
_THIS_FILE = pathlib.Path(__file__).resolve()

ROOT = None
for cand in [_THIS_FILE.parent, _THIS_FILE.parent.parent]:
    if (cand / "algorithms").exists():
        ROOT = cand
        break

if ROOT is None:
    raise RuntimeError("未找到项目根目录（包含 algorithms/ 的目录）")

sys.path.append(str(ROOT))

from algorithms.subproblem.router import detect_problem_type, evaluate_subproblem


# =========================================================
# Exact global solver 配置 / 统计
# =========================================================
@dataclass
class ExactGlobalConfig:
    round_digits: int = 6
    bound_tol: float = 1e-9
    search_order: str = "standalone_profit_desc"   # 目前只实现这一种
    branch_one_first: bool = True                  # 先搜 x_i = 1，再搜 x_i = 0


@dataclass
class ExactGlobalStats:
    standalone_evals: int = 0
    nodes_visited: int = 0
    leaves_evaluated: int = 0
    pruned_by_bound: int = 0


# =========================================================
# 你的 current solver 配置
# 注意：为了避免 top-level import 直接触发 pyqpanda 依赖，
# 这些 import 都延迟到函数内部
# =========================================================
def make_quantum_config():
    from algorithms.master.quantum_master import QuantumMasterConfig
    from algorithms.quantum.qubo_builder import QUBOConfig
    from algorithms.quantum.qaoa_solver import QAOASolverConfig

    return QuantumMasterConfig(
        qubo_config=QUBOConfig(
            objective_scale=1.0,
            resource_penalty=10.0,  # 修改，原来是10.0
            hamming_penalty=0.5,  # 修改，原来是0.5
            conflict_weight=1.0,
            tabu_penalty=5.0,
            elite_weight=0.5,  # 修改，原来是0.5
            exploration_boost=0.5,  # 修改，原来是0.5
        ),
        qaoa_config=QAOASolverConfig(
            p=1,
            shots=128,  # 修改，原来是128
            gamma_values=np.array([0.2, 0.5], dtype=float),  # 修改，原来是[0.2, 0.5]
            beta_values=np.array([0.2, 0.5], dtype=float),  # 修改，原来是[0.2, 0.5]
            seed=42,
            per_run_top_k=5,  # 修改，原来是5
            final_top_k=10,  # 修改，原来是10
        ),
        round_digits=6,
        candidate_top_k=8,  # 修改，原来是8
        include_solver_best=True,
        deduplicate_candidates=True,
        add_mutations_from_best=True,
        add_mutations_from_incumbent=True,
        max_best_mutations=3,  # 修改，原来是3
        max_incumbent_mutations=3,  # 修改，原来是3
    )


def make_iterative_hybrid_config():
    from algorithms.master.hybrid_master import HybridMasterConfig
    from algorithms.master.iterative_hybrid_master import IterativeHybridMasterConfig
    from algorithms.candidate.candidate_selector import CandidateSelectionConfig
    from algorithms.feedback.bias_updater import BiasUpdateConfig
    from algorithms.local_search.refiner import LocalRefineConfig

    iterative_cfg = IterativeHybridMasterConfig(
        use_classical_warm_start=False,
        max_rounds=3,  # 修改，原来是3
        no_improve_patience=2,  # 修改，原来是2
        classical_num_starts=5,  # 修改，原来是5
        classical_local_iter=20,  # 修改，原来是20
        classical_seed=42,
        quantum_config=make_quantum_config(),
        candidate_config=CandidateSelectionConfig(
            top_k=8,  # 修改，原来是8
            min_hamming_distance=2,
            prefer_higher_probability=True,
            min_far_hamming_distance=3,
            min_far_candidates=2,  # 修改，原来是2
            tabu_hamming_threshold=1,
            objective_bonus=0.05,
            far_bonus=0.02,  # 修改，原来是0.02
        ),
        bias_update_config=BiasUpdateConfig(),  # 修改，原来是空
        local_refine_config=LocalRefineConfig(
            max_iter=20,  # 修改，原来是20
            enable_swap=True,
            max_start_points=3,  # 修改，原来是3
            use_guided_order=True,
        ),
    )

    return HybridMasterConfig(
        use_classical=False,
        use_quantum=True,
        classical_num_starts=5,  # 修改，原来是5
        classical_local_iter=20,  # 修改，原来是20
        classical_seed=42,
        quantum_config=make_quantum_config(),
        fallback_to_classical=True,
        iterative=True,
        iterative_config=iterative_cfg,
    )

# =========================================================
# 跑 current solver
# =========================================================
def solve_with_current_solver(problem_dict: dict, mode: str = "iterative_hybrid") -> dict[str, Any]:
    from algorithms.pipeline.milp_pipeline import solve_milp
    from algorithms.pipeline.micp_pipeline import solve_micp

    problem_type = detect_problem_type(problem_dict)

    if mode == "iterative_hybrid":
        master_config = make_iterative_hybrid_config()
    else:
        raise ValueError(f"当前 compare 脚本只支持 mode=iterative_hybrid，收到: {mode}")

    if problem_type == "MILP":
        result = solve_milp(
            problem_dict,
            master_mode=mode,
            master_config=master_config,
        )
    else:
        result = solve_micp(
            problem_dict,
            master_mode=mode,
            master_config=master_config,
        )

    return result


# =========================================================
# exact global solver: 预计算单产品 standalone 最优利润
# 用来构造 BnB 上界
#
# 解释：
# 对每个 i，令 x=e_i（只有第 i 个产品打开），调用项目统一 evaluator，
# 得到该产品单独存在时的最优利润 Z_i^standalone。
#
# 则对任何联合解，其每个产品 i 的实际贡献都不可能超过 standalone 情况，
# 因而：
#     总目标 <= 所有被允许保留的产品的 standalone 正利润之和
#
# 这是一个有效（但不一定很紧）的全局上界。
# =========================================================
def compute_standalone_profits(
    problem_dict: dict,
    cfg: ExactGlobalConfig,
    stats: ExactGlobalStats,
) -> np.ndarray:
    n = int(problem_dict["product_count"])
    standalone = np.zeros(n, dtype=float)

    for i in range(n):
        x = np.zeros(n, dtype=float)
        x[i] = 1.0
        res = evaluate_subproblem(problem_dict, x, round_digits=cfg.round_digits)
        standalone[i] = float(res["Z"])
        stats.standalone_evals += 1

    return standalone


# =========================================================
# exact global solver: Branch-and-Bound over x
# 叶子节点调用项目的精确子问题 evaluator
# =========================================================
def solve_with_exact_global_opt(
    problem_dict: dict,
    cfg: ExactGlobalConfig | None = None,
    warm_start_solution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if cfg is None:
        cfg = ExactGlobalConfig()

    t0 = time.perf_counter()
    n = int(problem_dict["product_count"])
    stats = ExactGlobalStats()

    # ---------- 1) 单产品 standalone 利润 ----------
    standalone = compute_standalone_profits(problem_dict, cfg, stats)
    standalone_pos = np.maximum(standalone, 0.0)

    if cfg.search_order != "standalone_profit_desc":
        raise ValueError(f"暂不支持的 search_order: {cfg.search_order}")

    order = np.argsort(-standalone_pos)  # 按正 standalone 利润降序分枝
    total_positive_upper = float(np.sum(standalone_pos))

    # ---------- 2) 初始 incumbent（可用 current solver 做 warm start） ----------
    if warm_start_solution is not None:
        best_sol = {
            "x": list(warm_start_solution["x"]),
            "y": list(warm_start_solution["y"]),
            "Z": float(warm_start_solution["Z"]),
            "r": list(warm_start_solution["r"]),
        }
        best_z = float(warm_start_solution["Z"])
        best_source = "warm_start_from_current_solver"
    else:
        x0 = np.zeros(n, dtype=float)
        res0 = evaluate_subproblem(problem_dict, x0, round_digits=cfg.round_digits)
        best_sol = {
            "x": res0["x"],
            "y": res0["y"],
            "Z": float(res0["Z"]),
            "r": res0["r"],
        }
        best_z = float(res0["Z"])
        best_source = "all_zero_init"

    # ---------- 3) DFS + BnB ----------
    x_work = np.zeros(n, dtype=int)

    def dfs(depth: int, ub_active: float) -> None:
        nonlocal best_sol, best_z

        stats.nodes_visited += 1

        # 剪枝：该节点下所有可能 completion 的上界都不超过当前 incumbent
        if ub_active <= best_z + cfg.bound_tol:
            stats.pruned_by_bound += 1
            return

        # 到叶子：精确评估当前 x
        if depth == n:
            stats.leaves_evaluated += 1
            x_leaf = x_work.astype(float)
            res = evaluate_subproblem(problem_dict, x_leaf, round_digits=cfg.round_digits)
            z = float(res["Z"])

            if z > best_z + cfg.bound_tol:
                best_z = z
                best_sol = {
                    "x": res["x"],
                    "y": res["y"],
                    "Z": z,
                    "r": res["r"],
                }
            return

        i = int(order[depth])
        pos_i = float(standalone_pos[i])

        # 分枝顺序：通常先搜 x_i = 1，更容易更快拿到强 incumbent
        if cfg.branch_one_first:
            branch_values = (1, 0)
        else:
            branch_values = (0, 1)

        for val in branch_values:
            x_work[i] = val

            # 如果把 i 固定成 0，则这个产品的正 standalone 上界要从后续可用上界里扣掉
            if val == 1:
                child_ub = ub_active
            else:
                child_ub = ub_active - pos_i

            dfs(depth + 1, child_ub)

        # 恢复（虽然下一轮会覆盖，但保留一下更清楚）
        x_work[i] = 0

    dfs(depth=0, ub_active=total_positive_upper)

    runtime = time.perf_counter() - t0
    return {
        "status": "optimal",
        "objective_value": float(best_sol["Z"]),
        "solution": best_sol,
        "runtime": runtime,
        "extra": {
            "problem_type": detect_problem_type(problem_dict),
            "solver": "exact_branch_and_bound_over_x",
            "best_source": best_source,
            "standalone_profit": np.round(standalone, cfg.round_digits).tolist(),
            "branch_order": order.astype(int).tolist(),
            "stats": {
                "standalone_evals": stats.standalone_evals,
                "nodes_visited": stats.nodes_visited,
                "leaves_evaluated": stats.leaves_evaluated,
                "pruned_by_bound": stats.pruned_by_bound,
            },
        },
    }


# =========================================================
# 打印
# =========================================================
def print_solution_block(title: str, sol: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print("x:", sol["x"])
    print("y:", sol["y"])
    print("Z:", sol["Z"])
    print("r:", sol["r"])


# =========================================================
# compare
# =========================================================
def compare(problem_path: pathlib.Path) -> None:
    with open(problem_path, "r", encoding="utf-8") as f:
        problem_dict = json.load(f)

    print("=== Problem Info ===")
    print("file:", problem_path)
    print("type:", detect_problem_type(problem_dict))
    print("n(product_count):", problem_dict["product_count"])
    print("m(resource_count):", problem_dict["resource_count"])

    # ---------- current solver ----------
    current_result = solve_with_current_solver(problem_dict, mode="iterative_hybrid")
    current_sol = current_result["solution"]

    # ---------- exact global optimum ----------
    exact_result = solve_with_exact_global_opt(
        problem_dict,
        cfg=ExactGlobalConfig(
            round_digits=6,
            bound_tol=1e-9,
            search_order="standalone_profit_desc",
            branch_one_first=True,
        ),
        warm_start_solution=current_sol,   # 用 current solver 结果做 lower bound，加速精确搜索
    )
    exact_sol = exact_result["solution"]

    print_solution_block("Current Solver", current_sol)
    print_solution_block("Exact Global Optimum", exact_sol)

    z_current = float(current_sol["Z"])
    z_opt = float(exact_sol["Z"])

    abs_gap = z_opt - z_current
    rel_gap = abs_gap / max(1.0, abs(z_opt))
    hit_opt = abs(abs_gap) <= 1e-6

    x_current = np.asarray(current_sol["x"], dtype=float)
    x_opt = np.asarray(exact_sol["x"], dtype=float)
    same_x = bool(np.allclose(x_current, x_opt, atol=1e-9))

    print("\n=== Comparison Summary ===")
    print(f"Current solver Z : {z_current:.6f}")
    print(f"Global optimum Z : {z_opt:.6f}")
    print(f"Absolute gap     : {abs_gap:.6f}")
    print(f"Relative gap     : {100.0 * rel_gap:.4f}%")
    print(f"Hit optimum?     : {hit_opt}")
    print(f"Same x as opt?   : {same_x}")

    print("\n=== Runtime ===")
    print(f"Current solver runtime : {float(current_result.get('runtime', 0.0)):.6f} s")
    print(f"Exact solver runtime   : {float(exact_result.get('runtime', 0.0)):.6f} s")

    stats = exact_result["extra"]["stats"]
    print("\n=== Exact Solver Stats ===")
    print("standalone_evals:", stats["standalone_evals"])
    print("nodes_visited   :", stats["nodes_visited"])
    print("leaves_evaluated:", stats["leaves_evaluated"])
    print("pruned_by_bound :", stats["pruned_by_bound"])


def main():
    parser = argparse.ArgumentParser(
        description="Compare current solver with exact global optimum"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="问题 JSON 路径，例如 data/raw/problem_micp_1.json",
    )
    args = parser.parse_args()

    compare(pathlib.Path(args.input))


if __name__ == "__main__":
    main()