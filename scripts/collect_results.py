# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
PARAM_COMP_ROOT = ROOT / "output" / "param_comp"
FIG_ROOT = PARAM_COMP_ROOT / "fig"
BASE_CONFIG_PATH = ROOT / "config" / "iterative_hybrid_base.json"


# =========================
# 基础工具
# =========================
def load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"JSON 顶层必须是 object: {path}")
    return data


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def stable_str(v: Any) -> str:
    if isinstance(v, float):
        if math.isfinite(v):
            return f"{v:.10g}"
        return str(v)
    if isinstance(v, (list, tuple)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def parse_result_filename(path: pathlib.Path) -> tuple[str, str]:
    """
    文件名约定：
        <config_name>__<problem_name>.json
    """
    stem = path.stem
    if "__" not in stem:
        return stem, "unknown_problem"
    config_name, problem_name = stem.rsplit("__", 1)
    return config_name, problem_name


def short_param_name(full_path: str) -> str:
    """
    iterative_config.bias_update_config.reward_step -> reward_step
    quantum_config.qubo_config.hamming_penalty -> hamming_penalty
    """
    return full_path.split(".")[-1]


def try_parse_numeric(v: Any) -> tuple[bool, float | None]:
    try:
        return True, float(v)
    except Exception:
        return False, None


def normalize_problem_name(name: str) -> str:
    """
    尽量把 problem_milp_1 / milp1 统一成更短的标识
    """
    s = name.lower().replace("-", "_")
    mapping = {
        "problem_milp_1": "milp1",
        "problem_milp_2": "milp2",
        "problem_micp_1": "micp1",
        "problem_micp_2": "micp2",
        "milp1": "milp1",
        "milp2": "milp2",
        "micp1": "micp1",
        "micp2": "micp2",
    }
    return mapping.get(s, s)


# =========================
# 参数变化检测
# =========================
def detect_changed_params(
    base_cfg: dict[str, Any],
    cfg_snapshot: dict[str, Any],
) -> list[tuple[str, Any, Any]]:
    base_flat = flatten_dict(base_cfg)
    cfg_flat = flatten_dict(cfg_snapshot)

    all_keys = sorted(set(base_flat) | set(cfg_flat))
    diffs: list[tuple[str, Any, Any]] = []

    for key in all_keys:
        b = base_flat.get(key, None)
        c = cfg_flat.get(key, None)
        if b != c:
            diffs.append((key, b, c))

    return diffs


# =========================
# 收集结果
# =========================
def collect_rows(
    result_root: pathlib.Path,
    base_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if not result_root.exists():
        raise FileNotFoundError(f"结果目录不存在: {result_root}")

    module_dirs = [
        p for p in sorted(result_root.iterdir())
        if p.is_dir() and p.name != "fig"
    ]

    for module_dir in module_dirs:
        module_name = module_dir.name

        for result_path in sorted(module_dir.glob("*.json")):
            try:
                result = load_json(result_path)
            except Exception as e:
                print(f"[WARN] 跳过无法读取的结果文件: {result_path} | {e}")
                continue

            extra = result.get("extra", {})
            config_snapshot = extra.get("config_snapshot", {})
            config_name_from_extra = extra.get("config_name")

            config_name, problem_name = parse_result_filename(result_path)
            if config_name_from_extra:
                config_name = str(pathlib.Path(config_name_from_extra).stem)

            problem_name = normalize_problem_name(problem_name)

            changed = detect_changed_params(base_cfg, config_snapshot) if config_snapshot else []

            # 理想情况下每个配置只变 1 个参数
            if len(changed) == 1:
                changed_path, base_value, new_value = changed[0]
            else:
                changed_path = "|".join(x[0] for x in changed)
                base_value = "|".join(stable_str(x[1]) for x in changed)
                new_value = "|".join(stable_str(x[2]) for x in changed)

            solution = result.get("solution", {})

            row = {
                "module": module_name,
                "problem": problem_name,
                "config_name": config_name,
                "result_file": str(result_path.relative_to(ROOT)),
                "status": result.get("status"),
                "objective_value": safe_float(result.get("objective_value")),
                "solution_Z": safe_float(solution.get("Z")),
                "runtime_pipeline": safe_float(result.get("runtime")),
                "runtime_total": safe_float(extra.get("total_runtime")),
                "changed_count": len(changed),
                "changed_param_path": changed_path,
                "changed_param": short_param_name(changed_path) if "|" not in changed_path else changed_path,
                "base_value": stable_str(base_value),
                "param_value": stable_str(new_value),
                "config_file": extra.get("config_file"),
            }
            rows.append(row)

    return rows


# =========================
# CSV 输出
# =========================
def write_csv(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(f"[OK] CSV saved: {path}")


def build_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    long_df = df.copy().sort_values(
        by=["problem", "module", "changed_param", "config_name"],
        kind="stable",
    )
    tables["results_long"] = long_df

    ranking_df = df.copy()
    ranking_df["rank_in_problem_module_param"] = (
        ranking_df.groupby(["problem", "module", "changed_param"])["objective_value"]
        .rank(method="dense", ascending=False)
    )
    ranking_df = ranking_df.sort_values(
        by=["problem", "module", "changed_param", "rank_in_problem_module_param", "config_name"],
        kind="stable",
    )
    tables["results_ranked"] = ranking_df

    best_df = (
        ranking_df.sort_values(
            by=["problem", "module", "changed_param", "objective_value", "runtime_pipeline"],
            ascending=[True, True, True, False, True],
            kind="stable",
        )
        .groupby(["problem", "module", "changed_param"], as_index=False)
        .first()
    )
    tables["best_per_problem_module_param"] = best_df

    return tables


# =========================
# 绘图
# =========================
def plot_problem_module_param(df: pd.DataFrame, fig_root: pathlib.Path) -> None:
    """
    每张图只画：
        一个 problem
        一个 module
        一个 changed_param
    横轴：参数值
    纵轴：objective_value
    """
    grouped = df.groupby(["problem", "module", "changed_param"], dropna=False)

    for (problem, module, changed_param), g in grouped:
        if not changed_param:
            continue
        if "|" in str(changed_param):
            # 不是单参数改动，跳过
            continue

        if len(g) < 2:
            # 只有一个点没必要画
            continue

        x_numeric_ok = True
        x_vals_num: list[float] = []
        for x in g["param_value"]:
            ok, num = try_parse_numeric(x)
            if not ok or num is None:
                x_numeric_ok = False
                break
            x_vals_num.append(num)

        if x_numeric_ok:
            plot_df = g.copy()
            plot_df["_x"] = x_vals_num
            plot_df = plot_df.sort_values(by="_x", kind="stable")

            plt.figure(figsize=(8, 5))
            plt.plot(plot_df["_x"], plot_df["objective_value"], marker="o")
            plt.xlabel(str(changed_param))
        else:
            plot_df = g.copy().sort_values(by="config_name", kind="stable")
            plot_df["_x"] = list(range(len(plot_df)))

            plt.figure(figsize=(10, 5))
            plt.plot(plot_df["_x"], plot_df["objective_value"], marker="o")
            plt.xticks(plot_df["_x"], plot_df["param_value"], rotation=30, ha="right")
            plt.xlabel(str(changed_param))

        plt.ylabel("objective_value")
        plt.title(f"{problem} | {module} | {changed_param}")
        plt.grid(True, alpha=0.3)

        out_dir = fig_root / problem / module
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{changed_param}.png"

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[OK] Figure saved: {out_path}")


# =========================
# 主流程
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect parameter comparison results and draw problem-specific figures."
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default=str(PARAM_COMP_ROOT),
        help="结果目录，默认 output/param_comp",
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default=str(BASE_CONFIG_PATH),
        help="基准配置文件，默认 config/iterative_hybrid_base.json",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="是否绘图，默认不绘图",
    )
    args = parser.parse_args()

    result_root = pathlib.Path(args.result_root)
    base_cfg = load_json(pathlib.Path(args.base_config))

    rows = collect_rows(result_root=result_root, base_cfg=base_cfg)
    if not rows:
        print("[WARN] 未发现任何结果文件。")
        return

    df = pd.DataFrame(rows)
    tables = build_tables(df)

    write_csv(tables["results_long"], result_root / "results_long.csv")
    write_csv(tables["results_ranked"], result_root / "results_ranked.csv")
    write_csv(
        tables["best_per_problem_module_param"],
        result_root / "best_per_problem_module_param.csv",
    )

    if args.plot:
        plot_problem_module_param(tables["results_long"], FIG_ROOT)

    print("\nDone.")


if __name__ == "__main__":
    main()