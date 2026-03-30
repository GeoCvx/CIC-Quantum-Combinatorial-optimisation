import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from main import solve


def main():
    path = ROOT / "data" / "raw" / "problem_milp_1.json"
    result = solve(str(path))
    print(result)


if __name__ == "__main__":
    main()