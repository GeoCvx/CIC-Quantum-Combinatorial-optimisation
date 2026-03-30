import numpy as np

from algorithms.subproblem.router import evaluate_subproblem


def random_binary_x(n: int, rng: np.random.Generator):
    return rng.integers(0, 2, size=n).astype(float)


def search_x_random(problem_dict: dict, num_trials: int = 50, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = problem_dict["product_count"]

    best_x = None
    best_result = None
    best_z = -float("inf")

    for _ in range(num_trials):
        x = random_binary_x(n, rng)

        try:
            result = evaluate_subproblem(problem_dict, x, round_digits=6)
            z = result["Z"]
        except Exception:
            continue

        if z > best_z:
            best_z = z
            best_x = x.copy()
            best_result = result

    return {
        "best_x": best_x,
        "best_result": best_result,
        "best_objective": best_z,
    }