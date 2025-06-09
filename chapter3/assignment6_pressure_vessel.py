"""Task 6: Pressure vessel design using PSO."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import japanize_matplotlib  # type: ignore

from chapter2.pso import run_pso


def objective(x):
    x1, x2, x3, x4 = x
    return (
        0.6224 * x1 * x3 * x4
        + 1.7781 * x2 * x3 ** 2
        + 3.1661 * x1 ** 2 * x4
        + 19.84 * x1 ** 2 * x3
    )


def constraints(x):
    x1, x2, x3, x4 = x
    g1 = -x1 + 0.0193 * x3
    g2 = -x2 + 0.00954 * x3
    g3 = -np.pi * x3 ** 2 * x4 - (4 / 3) * np.pi * x3 ** 3 + 1296000
    g4 = x4 - 240
    return np.array([g1, g2, g3, g4])


def penalty_function(x):
    penalty_factor = 1e6
    pen = np.sum(np.maximum(0, constraints(x))) * penalty_factor
    return objective(x) + pen


BOUNDS = [
    (0.0625, 6.1875),
    (0.0625, 6.1875),
    (10, 200),
    (10, 200),
]
DISCRETE = {0, 1}


def clip_position(pos):
    new = []
    for i, (low, high) in enumerate(BOUNDS):
        val = min(max(pos[i], low), high)
        if i in DISCRETE:
            val = round(val / 0.0625) * 0.0625
        new.append(val)
    return new


def pso_run():
    num_particles = 30
    iterations = 200
    dim = 4
    x_min = [b[0] for b in BOUNDS]
    x_max = [b[1] for b in BOUNDS]
    v_min = -1
    v_max = 1

    positions = [
        [np.random.uniform(l, h) for l, h in BOUNDS]
        for _ in range(num_particles)
    ]
    for p in positions:
        for i in DISCRETE:
            p[i] = round(p[i] / 0.0625) * 0.0625

    velocities = [[np.random.uniform(v_min, v_max) for _ in range(dim)] for _ in range(num_particles)]
    pbest_pos = [p.copy() for p in positions]
    pbest_val = [penalty_function(p) for p in positions]
    gbest_index = int(np.argmin(pbest_val))
    gbest_pos = positions[gbest_index].copy()
    gbest_val = pbest_val[gbest_index]
    history = []

    for t in range(iterations):
        for p in range(num_particles):
            for d in range(dim):
                positions[p][d] += velocities[p][d]
            positions[p] = clip_position(positions[p])
            val = penalty_function(positions[p])
            if val < pbest_val[p]:
                pbest_val[p] = val
                pbest_pos[p] = positions[p].copy()
            if val < gbest_val:
                gbest_val = val
                gbest_pos = positions[p].copy()
        history.append(gbest_val)
        for p in range(num_particles):
            for d in range(dim):
                r1 = np.random.rand()
                r2 = np.random.rand()
                velocities[p][d] = (
                    0.5 * velocities[p][d]
                    + 2 * r1 * (pbest_pos[p][d] - positions[p][d])
                    + 2 * r2 * (gbest_pos[d] - positions[p][d])
                )
    return gbest_pos, gbest_val, history


def main() -> None:
    trials = 10
    histories = []
    best_positions = []
    best_values = []
    for _ in range(trials):
        pos, val, hist = pso_run()
        histories.append(hist)
        best_positions.append(pos)
        best_values.append(val)

    for hist in histories:
        plt.plot(hist)
    plt.xlabel("世代（繰り返し回数）")
    plt.ylabel("目的関数値")
    plt.yscale("log")
    plt.show()

    df = pd.DataFrame({
        "T_s": [p[0] for p in best_positions],
        "T_h": [p[1] for p in best_positions],
        "R": [p[2] for p in best_positions],
        "L": [p[3] for p in best_positions],
        "目的関数値": best_values,
    })
    print(df)


if __name__ == "__main__":
    main()
