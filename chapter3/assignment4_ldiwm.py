"""Task 4: Implement PSO with linearly decreasing inertia weight."""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import japanize_matplotlib  # type: ignore

# Add chapter2 directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "chapter2"))

from pso import run_pso, rastrigin


def main() -> None:
    _, best_val, history = run_pso(
        rastrigin,
        num_particles=20,
        dim=20,
        iterations=1000,
        x_min=-5.12,
        x_max=5.12,
        v_min=-1,
        v_max=1,
        ldiwm=True,
        w_min=0.4,
        w_max=0.9,
    )
    print("Best", best_val)
    plt.plot(history)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Best value")
    plt.show()


if __name__ == "__main__":
    main()
