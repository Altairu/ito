import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# Rastrigin関数の定義
def fitFunc1(xVals):
    fitness = 10 * len(xVals)
    for x in xVals:
        fitness += x**2 - 10 * math.cos(2 * math.pi * x)
    return fitness

# 粒子の位置情報の初期化
def initPosition(Np, Nd, xMin, xMax):
    return [[xMin + random.random() * (xMax - xMin) for _ in range(Nd)] for _ in range(Np)]

# 粒子の速度情報の初期化
def initVelocity(Np, Nd, vMin, vMax):
    return [[vMin + random.random() * (vMax - vMin) for _ in range(Nd)] for _ in range(Np)]

# 粒子の位置ベクトルの更新
def updatePosition(R, V, Np, Nd, xMin, xMax):
    for p in range(Np):
        for i in range(Nd):
            R[p][i] += V[p][i]
            R[p][i] = max(min(R[p][i], xMax), xMin)

# 粒子の速度ベクトルの更新
def updateVelocity(R, V, Np, Nd, j, w, vMin, vMax, pBestPos, gBestPos, c1, c2):
    for p in range(Np):
        for i in range(Nd):
            r1 = random.random()
            r2 = random.random()
            V[p][i] = (w * V[p][i] +
                       r1 * c1 * (pBestPos[p][i] - R[p][i]) +
                       r2 * c2 * (gBestPos[i] - R[p][i]))
            V[p][i] = max(min(V[p][i], vMax), vMin)

# 粒子の評価値の更新
def updateFitness(R, M, Np, pBestPos, pBestVal, gBestPos, gBestValue):
    for p in range(Np):
        M[p] = fitFunc1(R[p])
        if M[p] < gBestValue:
            gBestValue = M[p]
            gBestPos[:] = R[p][:]
        if M[p] < pBestVal[p]:
            pBestVal[p] = M[p]
            pBestPos[p][:] = R[p][:]
    return gBestValue

def run_experiment(c1, c2, w_max, w_min, ITR=10):
    Np, Nd, Nt = 20, 20, 1000
    xMin, xMax = -500, 500
    vMin, vMax = 0.25 * xMin, 0.25 * xMax
    history_all = np.empty((ITR, Nt))
    for itr in range(ITR):
        R = initPosition(Np, Nd, xMin, xMax)
        V = initVelocity(Np, Nd, vMin, vMax)
        M = [fitFunc1(r) for r in R]
        pBestVal = M[:]
        pBestPos = [r[:] for r in R]
        gBestValue = min(M)
        gBestPos = R[M.index(gBestValue)][:]
        for j in range(Nt):
            updatePosition(R, V, Np, Nd, xMin, xMax)
            gBestValue = updateFitness(R, M, Np, pBestPos, pBestVal, gBestPos, gBestValue)
            # 線形減衰による慣性項の更新
            w = w_max - ((w_max - w_min) / Nt) * j
            updateVelocity(R, V, Np, Nd, j, w, vMin, vMax, pBestPos, gBestPos, c1, c2)
            history_all[itr][j] = gBestValue
    return np.mean(history_all, axis=0)

def main():
    # 複数パラメータ設定による比較実験
    param_sets = {
        "Standard":    {"c1": 2.05, "c2": 2.05, "w_max": 0.75, "w_min": 0.4},
        "Low_acc":     {"c1": 1.0,  "c2": 1.0,  "w_max": 0.75, "w_min": 0.4},
        "High_acc":    {"c1": 3.0,  "c2": 3.0,  "w_max": 0.75, "w_min": 0.4},
        "High_inertia":{"c1": 2.05, "c2": 2.05, "w_max": 0.9,  "w_min": 0.65},
        "Low_inertia": {"c1": 2.05, "c2": 2.05, "w_max": 0.6,  "w_min": 0.3},
    }
    results = {}
    for key, params in param_sets.items():
        print(f"{key} パラメータで実験中...")
        results[key] = run_experiment(**params)
    
    output_dir = "実験2結果考察"
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.index.name = "世代"
    df.to_csv(os.path.join(output_dir, "kadai2_comparison_history.csv"))
    
    plt.figure(figsize=(9,6))
    for key, history in results.items():
        plt.plot(history, label=key)
    plt.xlabel("世代", size=16)
    plt.ylabel("最良適応度", size=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kadai2_comparison_graph.png"))
    plt.show()

if __name__ == "__main__":
    main()