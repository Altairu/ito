"""Task 3: Compare performance on easy vs difficult problems."""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # type: ignore

from chapter2.pso import run_pso, sphere, rastrigin


def main() -> None:
    _, _, hist_sphere = run_pso(
        sphere, num_particles=20, dim=5, iterations=200, x_min=-5, x_max=5, v_min=-1, v_max=1
    )
    _, _, hist_rastrigin = run_pso(
        rastrigin, num_particles=20, dim=5, iterations=200, x_min=-5.12, x_max=5.12, v_min=-1, v_max=1
    )

    plt.plot(hist_sphere, label="Sphere")
    plt.plot(hist_rastrigin, label="Rastrigin")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Best value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
