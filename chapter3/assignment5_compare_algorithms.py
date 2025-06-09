"""Task 5: Compare standard PSO and LDIWM variant."""

import matplotlib.pyplot as plt
import japanize_matplotlib  # type: ignore

from chapter2.pso import run_pso, rastrigin


def run(method_name: str, ldiwm: bool) -> list[float]:
    _, _, history = run_pso(
        rastrigin,
        num_particles=20,
        dim=20,
        iterations=1000,
        x_min=-5.12,
        x_max=5.12,
        v_min=-1,
        v_max=1,
        ldiwm=ldiwm,
    )
    return history


def main() -> None:
    h_standard = run("standard", False)
    h_ldiwm = run("LDIWM", True)

    plt.plot(h_standard, label="standard")
    plt.plot(h_ldiwm, label="LDIWM")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Best value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
