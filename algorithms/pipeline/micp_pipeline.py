import time

from algorithms.master.classical_master import search_x_random


def solve_micp(problem_dict: dict, num_trials: int = 50):
    start = time.time()

    search_result = search_x_random(problem_dict, num_trials=num_trials)

    runtime = time.time() - start

    return {
        "status": "feasible",
        "objective_value": search_result["best_objective"],
        "solution": search_result["best_result"],
        "runtime": runtime,
        "extra": {
            "problem_type": "MICP",
            "num_trials": num_trials,
        },
    }