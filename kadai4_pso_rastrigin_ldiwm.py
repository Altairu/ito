# ライブラリのインポート
from random import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# Rastrigin関数の定義
def fitFunc1(xVals):
    fitness = 10 * len(xVals)
    for i in range(len(xVals)):
        fitness += xVals[i]**2 - (10 * math.cos(2 * math.pi * xVals[i]))
    return fitness

# Sphere関数の定義（必要に応じて追加）
def fitFunc2(xVals):
    return sum([x**2 for x in xVals])

# Schwefel関数の定義（必要に応じて追加）
def fitFunc3(xVals):
    return 418.9829 * len(xVals) - sum([x * math.sin(math.sqrt(abs(x))) for x in xVals])

# Griewank関数の定義（必要に応じて追加）
def fitFunc4(xVals):
    sum_part = sum([x**2/4000 for x in xVals])
    prod_part = 1
    for i in range(len(xVals)):
        prod_part *= math.cos(xVals[i]/math.sqrt(i+1))
    return sum_part - prod_part + 1

# 粒子の位置情報の初期化
def initPosition(Np, Nd, xMin, xMax):
    return [[xMin + random() * (xMax - xMin) for _ in range(Nd)] for _ in range(Np)]

# 粒子の移動方向の初期化
def initVelocity(Np, Nd, vMin, vMax):
    return [[vMin + random() * (vMax - vMin) for _ in range(Nd)] for _ in range(Np)]

# 粒子の速度ベクトルの更新
def updateVelocity(R, V, Np, Nd, j, w, vMin, vMax, pBestPos, gBestPos):
    for p in range(Np):
        for i in range(Nd):
            r1 = random()
            r2 = random()
            V[p][i] = w * V[p][i] + r1 * c1 * (pBestPos[p][i] - R[p][i]) + r2 * c2 * (gBestPos[i] - R[p][i])
            # 速度制限
            if V[p][i] > vMax:
                V[p][i] = vMax
            if V[p][i] < vMin:
                V[p][i] = vMin

# 粒子の位置ベクトルの更新
def updatePosition(R, Np, Nd, xMin, xMax):
    for p in range(Np):
        for i in range(Nd):
            R[p][i] = R[p][i] + V[p][i]
            # 定義域外なら補正
            if R[p][i] > xMax:
                R[p][i] = xMax
            if R[p][i] < xMin:
                R[p][i] = xMin

# 粒子の評価値の更新
def updateFitness(R, M, Np, pBestPos, pBestVal, gBestPos, gBestValue):
    for p in range(Np):
        M[p] = fitFunc1(R[p])
        if M[p] < gBestValue:
            gBestValue = M[p]
            gBestPos = R[p]
        if M[p] < pBestVal[p]:
            pBestVal[p] = M[p]
            pBestPos[p] = R[p]
    return gBestValue

# === one-shot 実行例 ===
if __name__ == "__main__":
    # パラメータ設定
    Np, Nd, Nt = 20, 20, 1000        # 粒子数，次元数，世代数
    c1, c2 = 2.05, 2.05             # 加速係数
    w = 0.75                       # 初期の慣性項（参照値）
    wMin, wMax = 0.4, 0.9          # LDIWM の場合の慣性項の最小値と最大値
    xMin, xMax = -5.12, 5.12       # 設計変数の定義域
    vMin, vMax = 0.25*xMin, 0.25*xMax  # 速度の制限

    # 初期化
    R = initPosition(Np, Nd, xMin, xMax)
    V = initVelocity(Np, Nd, vMin, vMax)
    M = [fitFunc1(R[p]) for p in range(Np)]
    pBestVal = M[:]
    pBestPos = [r[:] for r in R]
    gBestValue = min(M)
    gBestPos = R[M.index(gBestValue)][:]
    history = [gBestValue]

    # 1回の PSO 実行
    for j in range(Nt):
        updatePosition(R, Np, Nd, xMin, xMax)
        gBestValue = updateFitness(R, M, Np, pBestPos, pBestVal, gBestPos, gBestValue)
        w = wMax - ((wMax - wMin) / Nt) * j  # LDIWM による慣性項の更新
        updateVelocity(R, V, Np, Nd, j, w, vMin, vMax, pBestPos, gBestPos)
        history.append(gBestValue)

    # 結果のグラフ表示と保存
    df = pd.DataFrame(history, columns=["最良評価値"])
    df.index.name = "世代"
    
    import os
    os.makedirs("実験4結果", exist_ok=True)
    df.to_csv("実験4結果/kadai4_result_history_fixed.csv")
    
    plt.figure(figsize=(9, 6))
    plt.plot(history, label="Rastrigin関数")
    plt.xlabel("世代", size=16)
    plt.ylabel("目的関数値", size=16)
    plt.legend()
    plt.grid(True)
    plt.savefig("実験4結果/kadai4_result_graph_fixed.png")
    plt.show()