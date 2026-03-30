import numpy as np

from algorithms.subproblem.milp_lp import evaluate_x as evaluate_lp_x
from algorithms.subproblem.micp_qp import evaluate_x as evaluate_qp_x


def detect_problem_type(problem_dict: dict, eps: float = 1e-12) -> str:
    beta = np.asarray(problem_dict["beta"], dtype=float)
    if np.all(np.abs(beta) <= eps):
        return "MILP"
    return "MICP"


def evaluate_subproblem(problem_dict: dict, x, round_digits: int = 2):
    """
    统一子问题入口：
    自动判断 MILP / MICP
    """
    problem_type = detect_problem_type(problem_dict)

    if problem_type == "MILP":
        return evaluate_lp_x(problem_dict, x, round_digits=round_digits)

    return evaluate_qp_x(problem_dict, x, round_digits=round_digits)