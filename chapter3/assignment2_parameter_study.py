"""Task 2: Evaluate PSO parameters on Rastrigin function."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib  # type: ignore

from chapter2.pso import run_pso, rastrigin


def main() -> None:
    Np, Nd, Nt = 20, 20, 1000
    c1 = c2 = 2.05
    w = 0.75
    xMin, xMax = -500, 500
    vMin, vMax = 0.25 * xMin, 0.25 * xMax

    history = np.empty((10, Nt))

    for i in range(10):
        _, _, h = run_pso(
            rastrigin,
            num_particles=Np,
            dim=Nd,
            iterations=Nt,
            x_min=xMin,
            x_max=xMax,
            v_min=vMin,
            v_max=vMax,
            c1=c1,
            c2=c2,
            inertia=w,
        )
        history[i] = h

    df = pd.DataFrame(history).T
    ax = df.plot(logy=True, xlim=[0, Nt], figsize=(9, 6))
    ax.set_xlabel("繰り返し回数")
    ax.set_ylabel("目的関数値")
    plt.show()


if __name__ == "__main__":
    main()
