import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# 日本語フォント設定（Noto CJKなどを使う）
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

# 目的関数：圧力容器のコスト
def objective_function(x):
    x1, x2, x3, x4 = x  # x1=T_s, x2=T_h, x3=R, x4=L
    return (
        0.6224 * x1 * x3 * x4 +
        1.7781 * x2 * x3 ** 2 +
        3.1661 * x1 ** 2 * x4 +
        19.84 * x1 ** 2 * x3
    )

# 制約関数（不等式制約g1〜g4 <= 0）
def constraints(x):
    x1, x2, x3, x4 = x
    g1 = -x1 + 0.0193 * x3
    g2 = -x2 + 0.00954 * x3
    g3 = -np.pi * x3**2 * x4 - (4/3)*np.pi * x3**3 + 1296000
    g4 = x4 - 240
    return np.array([g1, g2, g3, g4])

# ペナルティ付き目的関数
def penalty_function(x):
    penalty_factor = 1e6
    cons = constraints(x)
    penalty = np.sum(np.maximum(0, cons)) * penalty_factor
    return objective_function(x) + penalty

# PSOアルゴリズム
def pso(objective, bounds, discrete_indices, num_particles=30, max_iter=100, w=0.5, c1=2, c2=2):
    num_dimensions = len(bounds)
    positions = np.array([
        [np.random.uniform(low, high) for low, high in bounds]
        for _ in range(num_particles)
    ])
    
    for i in discrete_indices:
        positions[:, i] = np.round(positions[:, i] / 0.0625) * 0.0625

    velocities = np.zeros((num_particles, num_dimensions))
    personal_best_positions = positions.copy()
    personal_best_values = np.array([objective(pos) for pos in positions])
    global_best_position = positions[np.argmin(personal_best_values)]
    global_best_value = np.min(personal_best_values)

    best_values = []

    for iteration in range(max_iter):
        best_values.append(global_best_value)

        for i in range(num_particles):
            r1 = np.random.rand(num_dimensions)
            r2 = np.random.rand(num_dimensions)
            velocities[i] = (
                w * velocities[i] +
                c1 * r1 * (personal_best_positions[i] - positions[i]) +
                c2 * r2 * (global_best_position - positions[i])
            )
            positions[i] += velocities[i]
            for j, (low, high) in enumerate(bounds):
                positions[i, j] = np.clip(positions[i, j], low, high)
            for j in discrete_indices:
                positions[i, j] = np.round(positions[i, j] / 0.0625) * 0.0625

            current_value = objective(positions[i])
            if current_value < personal_best_values[i]:
                personal_best_positions[i] = positions[i].copy()
                personal_best_values[i] = current_value

        global_best_position = personal_best_positions[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)

    return global_best_position, global_best_value, best_values

# メイン処理
if __name__ == "__main__":
    os.makedirs("実験6結果", exist_ok=True)

    bounds = [(0.0625, 6.1875), (0.0625, 6.1875), (10, 200), (10, 200)]
    discrete_indices = [0, 1]
    num_trials = 10

    all_results = []
    best_positions = []
    best_values = []

    for trial in range(num_trials):
        best_pos, best_val, history = pso(
            penalty_function, bounds, discrete_indices,
            num_particles=30, max_iter=100
        )
        all_results.append(history)
        best_positions.append(best_pos)
        best_values.append(best_val)

    # グラフ描画
    plt.figure(figsize=(9, 6))
    for i, history in enumerate(all_results):
        plt.plot(range(1, len(history) + 1), history, label=f"試行 {i+1}")
    plt.xlabel("世代（繰り返し回数）", fontsize=14)
    plt.ylabel("目的関数値（コスト）", fontsize=14)
    plt.title("圧力容器設計最適化（PSOによる10試行）", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("実験6結果/kadai6_result_graph.png")
    plt.show()

    # 結果表
    df = pd.DataFrame({
        "試行": list(range(1, num_trials + 1)),
        "T_s": [round(p[0], 4) for p in best_positions],
        "T_h": [round(p[1], 4) for p in best_positions],
        "R":   [round(p[2], 4) for p in best_positions],
        "L":   [round(p[3], 4) for p in best_positions],
        "目的関数値": [round(v, 4) for v in best_values]
    })

    df.to_csv("実験6結果/kadai6_result_table.csv", index=False)
    print(df)
