import numpy as np
import matplotlib.pyplot as plt

# 日本語フォント設定（Noto CJKなどを使う）
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
# 材料密度（鋼材）
rho = 7.85e-6  # kg/mm^3

# 部材数
m = 10

# 部材長（仮定、全て1000mm）
lengths = np.ones(m) * 1000

# 許容応力（MPa）
sigma_allow = 150

# 外力（仮定：10kNが1方向にかかる）
external_force = np.ones(m) * 10000  # N

# 疑似的に応力を評価（実際は構造解析が必要）
def fake_stress(A):
    return external_force / A / 1e6  # [MPa]

# 目的関数（重量最小化）
def objective(A):
    return np.sum(rho * lengths * A)

# 制約付きペナルティ関数
def penalty_function(A):
    penalty = 0
    sigma = fake_stress(A)
    if np.any(sigma > sigma_allow):
        penalty += np.sum((sigma - sigma_allow) * 1e6)
    return objective(A) + penalty

# PSOアルゴリズム
def pso_truss(bounds, num_particles=30, max_iter=100, w=0.5, c1=2, c2=2):
    dim = len(bounds)
    pos = np.array([[np.random.uniform(low, high) for low, high in bounds] for _ in range(num_particles)])
    vel = np.zeros_like(pos)

    p_best_pos = pos.copy()
    p_best_val = np.array([penalty_function(p) for p in pos])
    g_best_pos = p_best_pos[np.argmin(p_best_val)]
    g_best_val = np.min(p_best_val)

    history = []

    for t in range(max_iter):
        history.append(g_best_val)
        for i in range(num_particles):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            vel[i] = (
                w * vel[i]
                + c1 * r1 * (p_best_pos[i] - pos[i])
                + c2 * r2 * (g_best_pos - pos[i])
            )
            pos[i] += vel[i]
            for d in range(dim):
                pos[i, d] = np.clip(pos[i, d], bounds[d][0], bounds[d][1])

            val = penalty_function(pos[i])
            if val < p_best_val[i]:
                p_best_pos[i] = pos[i].copy()
                p_best_val[i] = val
        g_best_pos = p_best_pos[np.argmin(p_best_val)]
        g_best_val = np.min(p_best_val)
    
    return g_best_pos, g_best_val, history

# 実行
if __name__ == "__main__":
    bounds = [(0.1, 35.0)] * m
    best_pos, best_val, hist = pso_truss(bounds)

    print("最適断面積:", best_pos)
    print("最小重量(kg):", best_val)

    # 収束グラフ
    plt.plot(hist)
    plt.xlabel("世代")
    plt.ylabel("重量 [kg]")
    plt.yscale("log")
    plt.title("トラス構造の重量最小化（PSO）")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
