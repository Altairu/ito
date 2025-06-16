# ライブラリのインポート
from random import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# Rastrigin関数の定義
def fitFunc1(xVals):
    # 初期値として10 * 次元数の値を設定
    fitness = 10 * len(xVals)
    for i in range(len(xVals)):
    # 各次元の値に対する計算を加算
        fitness += xVals[i]**2 - (10 * math.cos(2 * math.pi * xVals[i]))
    return fitness

# Sphere関数の定義
def fitFunc2(xVals):
    return sum([x**2 for x in xVals])

# Schwefel関数の定義
def fitFunc3(xVals):
    return 418.9829 * len(xVals) - sum([x * math.sin(math.sqrt(abs(x))) for x in xVals])

# Griewank関数の定義
def fitFunc4(xVals):
    sum_part = sum([x**2 / 4000 for x in xVals])
    prod_part = 1
    for i in range(len(xVals)):
        prod_part *= math.cos(xVals[i] / math.sqrt(i + 1))
    return sum_part - prod_part + 1


# 粒子の位置情報の初期化
def initPosition(Np, Nd, xMin, xMax):
    R = [[xMin + random() * (xMax - xMin) for i in range(0, Nd)] for p in range(0, Np)]
    return R

# 粒子の移動方向の初期化
def initVelocity(Np, Nd, vMin, vMax):
    V =[[vMin + random() * (vMax - vMin) for i in range(0, Nd)] for p in range(0, Np)]
    return V

# 粒子の速度ベクトルの更新
def updateVelocity(R, V, Np, Nd, j, w, vMin, vMax, pBestPos, gBestPos):
    for p in range(0, Np):
        for i in range(0, Nd):
            # ランダムな値r1, r2を生成
            r1 = random()
            r2 = random()
            # 速度ベクトルの更新
            V[p][i] = w * V[p][i] + r1 * c1 * (pBestPos[p][i] - R[p][i]) + r2 * c2 * (gBestPos[i] - R[p][i])
            # 速度制限
            if V[p][i] > vMax: V[p][i] = vMax
            if V[p][i] < vMin: V[p][i] = vMin

# 粒子の位置ベクトルの更新
def updatePosition(R, Np, Nd, xMin, xMax):
    for p in range(0, Np):
        for i in range(0, Nd):
            R[p][i] = R[p][i] + V[p][i]
            # 定義域外の場合には強制的に修正
            if R[p][i] > xMax: R[p][i] = xMax
            if R[p][i] < xMin: R[p][i] = xMin

# 粒子の評価値の更新
def updateFitness(R, M, Np, pBestPos, pBestVal, gBestPos, gBestValue):
    for p in range(0, Np):
        # 目的関数の評価
        M[p] = fitFunc1(R[p])
        # gBestの更新
        if M[p] < gBestValue:
            gBestValue = M[p]
            gBestPos = R[p]
        # pBestの更新
        if M[p] < pBestVal[p]:
            pBestVal[p] = M[p]
            pBestPos[p] = R[p]
        return gBestValue

if __name__ == "__main__":
    Np, Nd, Nt = 20, 20, 1000        # 粒子数，次元数，世代数
    c1, c2 = 2.05, 2.05             # 係数
    w = 0.75                      # 初期の慣性項（参照値）
    wMin, wMax = 0.4, 0.9         # LDIWM の場合の慣性項の最小値と最大値
    xMin, xMax = -5.12, 5.12      # 設計変数の定義域の最大・最小値
    vMin, vMax = 0.25*xMin, 0.25*xMax  # 速度ベクトルの最大・最小値

    # 試行回数 ITR と最良値を記録するためのリスト history
    ITR = 10  # 試行回数
    history = np.empty((ITR, Nt))

    for i in range(0, ITR):
        gBestValue = float("inf")                    # gBest（評価値）
        pBestValue = [float("inf")] * Np             # pBest（評価値）
        pBestPos = [[0] * Nd for _ in range(Np)]      # pBest の位置ベクトル
        gBestPos = [0] * Nd                          # gBest の位置ベクトル

        # 初期化
        R = initPosition(Np, Nd, xMin, xMax)
        V = initVelocity(Np, Nd, vMin, vMax)
        M = [fitFunc1(R[p]) for p in range(0, Np)]   # 目的関数

        for j in range(0, Nt):
            # 粒子の位置更新
            updatePosition(R, Np, Nd, xMin, xMax)
            # gBest の評価値を更新
            gBestValue = updateFitness(R, M, Np, pBestPos, pBestValue, gBestPos, gBestValue)
            # 履歴に記録
            history[i][j] = gBestValue
            # LDIWM: 慣性項を線形に減衰させる
            w = wMax - ((wMax - wMin) / Nt) * j
            # 速度の更新
            updateVelocity(R, V, Np, Nd, j, w, vMin, vMax, pBestPos, gBestPos)
    
    # グラフ表示
    df = pd.DataFrame(history).T
    df.plot(logy=True,
            xlim=(0, Nt),
            fontsize=14,
            figsize=(9, 6))
    
    import os
    os.makedirs("実験4結果", exist_ok=True)
    plt.savefig("実験4結果/kadai4_result_graph_fixed.png")
    df.to_csv("実験4結果/kadai4_result_history_fixed.csv", index_label="世代")

    plt.xlabel('繰り返し回数', size=16)
    plt.ylabel('目的関数値', size=16)
    plt.show()