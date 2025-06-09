"""Task 7: Example of another design problem optimized by PSO."""

import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib  # type: ignore

from chapter2.pso import run_pso

# Simple spring design problem parameters
# x1: wire diameter d (inch)
# x2: mean coil diameter D (inch)
# x3: number of active coils N


def spring_objective(x):
    d, D, N = x
    return (D + d) * N


BOUNDS = [(0.05, 2.0), (0.25, 5.0), (1.0, 15.0)]
DISCRETE = {2}


def clip(pos):
    new = []
    for i, (low, high) in enumerate(BOUNDS):
        val = min(max(pos[i], low), high)
        if i in DISCRETE:
            val = round(val)
        new.append(val)
    return new


def pso_run():
    num_particles = 30
    iterations = 200
    dim = 3
    positions = [[np.random.uniform(l, h) for l, h in BOUNDS] for _ in range(num_particles)]
    for p in positions:
        p[2] = round(p[2])
    velocities = [[np.random.uniform(-1, 1) for _ in range(dim)] for _ in range(num_particles)]
    pbest_pos = [p.copy() for p in positions]
    pbest_val = [spring_objective(p) for p in positions]
    gbest_idx = int(np.argmin(pbest_val))
    gbest_pos = positions[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]
    history = []

    for t in range(iterations):
        for i in range(num_particles):
            for d in range(dim):
                positions[i][d] += velocities[i][d]
            positions[i] = clip(positions[i])
            val = spring_objective(positions[i])
            if val < pbest_val[i]:
                pbest_val[i] = val
                pbest_pos[i] = positions[i].copy()
            if val < gbest_val:
                gbest_val = val
                gbest_pos = positions[i].copy()
        history.append(gbest_val)
        for i in range(num_particles):
            for d in range(dim):
                r1 = np.random.rand()
                r2 = np.random.rand()
                velocities[i][d] = (
                    0.5 * velocities[i][d]
                    + 2 * r1 * (pbest_pos[i][d] - positions[i][d])
                    + 2 * r2 * (gbest_pos[d] - positions[i][d])
                )
    return gbest_pos, gbest_val, history


def main() -> None:
    pos, val, hist = pso_run()
    print("Best", pos, val)
    plt.plot(hist)
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
