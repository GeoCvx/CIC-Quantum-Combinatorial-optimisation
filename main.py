import json
import numpy as np

from algorithms.pipeline.milp_pipeline import solve_milp
from algorithms.pipeline.micp_pipeline import solve_micp
from algorithms.subproblem.router import detect_problem_type


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def solve(path: str):
    problem_dict = load_json(path)
    problem_type = detect_problem_type(problem_dict)

    if problem_type == "MILP":
        return solve_milp(problem_dict)

    return solve_micp(problem_dict)


if __name__ == "__main__":
    path = "data/raw/problem_micp_1.json"
    result = solve(path)
    print(result)