import math
import random
from typing import Callable, List, Tuple


def rastrigin(x: List[float]) -> float:
    return 10 * len(x) + sum(xi ** 2 - 10 * math.cos(2 * math.pi * xi) for xi in x)


def sphere(x: List[float]) -> float:
    return sum(xi ** 2 for xi in x)


def rosenbrock(x: List[float]) -> float:
    return sum(100 * (x[i] - x[i - 1] ** 2) ** 2 + (x[i - 1] - 1) ** 2 for i in range(1, len(x)))


def griewank(x: List[float]) -> float:
    sum_term = sum(xi ** 2 for xi in x) / 4000.0
    prod_term = 1.0
    for i, xi in enumerate(x, start=1):
        prod_term *= math.cos(xi / math.sqrt(i))
    return sum_term - prod_term + 1


def init_position(num_particles: int, dim: int, x_min: float, x_max: float) -> List[List[float]]:
    return [[x_min + random.random() * (x_max - x_min) for _ in range(dim)] for _ in range(num_particles)]


def init_velocity(num_particles: int, dim: int, v_min: float, v_max: float) -> List[List[float]]:
    return [[v_min + random.random() * (v_max - v_min) for _ in range(dim)] for _ in range(num_particles)]


def update_velocity(
    positions: List[List[float]],
    velocities: List[List[float]],
    inertia: float,
    c1: float,
    c2: float,
    pbest_pos: List[List[float]],
    gbest_pos: List[float],
    v_min: float,
    v_max: float,
) -> None:
    for p in range(len(positions)):
        for i in range(len(positions[p])):
            r1 = random.random()
            r2 = random.random()
            velocities[p][i] = (
                inertia * velocities[p][i]
                + c1 * r1 * (pbest_pos[p][i] - positions[p][i])
                + c2 * r2 * (gbest_pos[i] - positions[p][i])
            )
            velocities[p][i] = max(min(velocities[p][i], v_max), v_min)


def update_position(
    positions: List[List[float]],
    velocities: List[List[float]],
    x_min: float,
    x_max: float,
) -> None:
    for p in range(len(positions)):
        for i in range(len(positions[p])):
            positions[p][i] += velocities[p][i]
            positions[p][i] = max(min(positions[p][i], x_max), x_min)


def update_fitness(
    func: Callable[[List[float]], float],
    positions: List[List[float]],
    pbest_pos: List[List[float]],
    pbest_val: List[float],
    gbest_pos: List[float],
    gbest_val: float,
) -> Tuple[List[List[float]], List[float], List[float], float]:
    for p in range(len(positions)):
        val = func(positions[p])
        if val < pbest_val[p]:
            pbest_val[p] = val
            pbest_pos[p] = positions[p].copy()
        if val < gbest_val:
            gbest_val = val
            gbest_pos = positions[p].copy()
    return pbest_pos, pbest_val, gbest_pos, gbest_val


def run_pso(
    func: Callable[[List[float]], float],
    num_particles: int,
    dim: int,
    iterations: int,
    x_min: float,
    x_max: float,
    v_min: float,
    v_max: float,
    c1: float = 2.05,
    c2: float = 2.05,
    inertia: float = 0.75,
    ldiwm: bool = False,
    w_min: float = 0.4,
    w_max: float = 0.9,
) -> Tuple[List[float], float, List[float]]:
    positions = init_position(num_particles, dim, x_min, x_max)
    velocities = init_velocity(num_particles, dim, v_min, v_max)
    pbest_pos = [pos.copy() for pos in positions]
    pbest_val = [func(pos) for pos in positions]
    gbest_index = pbest_val.index(min(pbest_val))
    gbest_pos = positions[gbest_index].copy()
    gbest_val = pbest_val[gbest_index]
    history = []

    for t in range(iterations):
        if ldiwm:
            inertia = w_max - ((w_max - w_min) / iterations) * t
        update_position(positions, velocities, x_min, x_max)
        pbest_pos, pbest_val, gbest_pos, gbest_val = update_fitness(
            func, positions, pbest_pos, pbest_val, gbest_pos, gbest_val
        )
        history.append(gbest_val)
        update_velocity(
            positions, velocities, inertia, c1, c2, pbest_pos, gbest_pos, v_min, v_max
        )
    return gbest_pos, gbest_val, history


if __name__ == "__main__":
    best_pos, best_val, hist = run_pso(rastrigin, 20, 20, 100, -5.12, 5.12, -1, 1)
    print("Best value", best_val)
