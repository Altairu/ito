"""Task 1: Benchmarking PSO with several test functions."""
from typing import List
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import japanize_matplotlib  # type: ignore

# Add chapter2 directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "chapter2"))

from pso import run_pso, sphere, rosenbrock, griewank, rastrigin


def optimize(func, title: str) -> None:
    best_pos, best_val, history = run_pso(
        func, num_particles=20, dim=20, iterations=200,
        x_min=-5.12, x_max=5.12, v_min=-1, v_max=1
    )
    print(title, "best value", best_val)
    plt.plot(history)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Best value")
    plt.yscale("log")
    plt.show()


def main() -> None:
    optimize(sphere, "Sphere")
    optimize(rosenbrock, "Rosenbrock")
    optimize(griewank, "Griewank")
    optimize(rastrigin, "Rastrigin")


if __name__ == "__main__":
    main()
