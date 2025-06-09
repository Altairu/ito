import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import japanize_matplotlib

def objective_function(x):
    x1, x2, x3, x4 = x
    return (
        0.6224 * x1 * x3 * x4 +
        1.7781 * x2 * x3**2 +
        3.1661 * x1**2 * x4 +
        19.84 * x1**2 * x3
    )

def constraints(x):
    x1, x2, x3, x4 = x
    return np.array([
        -x1 + 0.0193 * x3,
        -x2 + 0.00954 * x3,
        -np.pi * x3**2 * x4 - (4/3) * np.pi * x3**3 + 1296000,
        x4 - 240
    ])

def penalty_function(x):
    penalty_factor = 1e6
    constraint_values = constraints(x)
    penalty = np.sum(np.maximum(0, constraint_values)) * penalty_factor
    return objective_function(x) + penalty

# PSOアルゴリズムの実装
def pso(n_particles, n_iterations, bounds):
    # 粒子の初期化
    particles = np.random.rand(n_particles, len(bounds))
    velocities = np.random.rand(n_particles, len(bounds))
    personal_best_positions = np.copy(particles)
    personal_best_values = np.array([penalty_function(p) for p in particles])
    global_best_position = personal_best_positions[np.argmin(personal_best_values)]

    # 反復処理
    for _ in range(n_iterations):
        # 粒子の位置と速度を更新
        r1, r2 = np.random.rand(2)
        velocities = 0.5 * velocities + r1 * (personal_best_positions - particles) + r2 * (global_best_position - particles)
        particles += velocities

        # 位置の制約条件を適用
        particles = np.clip(particles, bounds[:, 0], bounds[:, 1])

        # 個々の粒子の最適値を更新
        fitness_values = np.array([penalty_function(p) for p in particles])
        better_mask = fitness_values < personal_best_values
        personal_best_positions[better_mask] = particles[better_mask]
        personal_best_values[better_mask] = fitness_values[better_mask]

        # グローバル最適値の更新
        global_best_position = personal_best_positions[np.argmin(personal_best_values)]

    return global_best_position, penalty_function(global_best_position)

if __name__ == "__main__":
    # パラメータ設定
    n_particles = 30
    n_iterations = 100
    bounds = np.array([[0, 100], [0, 100], [0, 100], [0, 100]])

    # PSO実行
    best_position, best_value = pso(n_particles, n_iterations, bounds)

    # 結果表示
    print("最適解:", best_position)
    print("最適値:", best_value)

    # 制約条件の表示
    con = constraints(best_position)
    print("制約条件の値:", con)
    print("ペナルティ項:", np.sum(np.maximum(0, con)) * 1e6)