import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Noto Sans CJK JP'

# --- Rastrigin関数 ---
def fitFunc1(xVals):
    A = 10
    return A * len(xVals) + sum(x**2 - A * np.cos(2 * np.pi * x) for x in xVals)

# --- 初期化関数 ---
def initPosition(Np, Nd, xMin, xMax):
    return np.random.uniform(low=xMin, high=xMax, size=(Np, Nd))

def initVelocity(Np, Nd, vMin, vMax):
    return np.random.uniform(low=vMin, high=vMax, size=(Np, Nd))

# --- 位置更新関数 ---
def updatePosition(R, V, xMin, xMax):
    R += V
    np.clip(R, xMin, xMax, out=R)

# --- 速度更新関数（wを受け取る） ---
def updateVelocity(R, V, w, vMin, vMax, pBestPos, gBestPos, c1, c2):
    Np, Nd = R.shape
    r1 = np.random.rand(Np, Nd)
    r2 = np.random.rand(Np, Nd)
    cognitive = c1 * r1 * (pBestPos - R)
    social = c2 * r2 * (gBestPos - R)
    V[:] = w * V + cognitive + social
    np.clip(V, vMin, vMax, out=V)

# --- 適応度と最良解更新 ---
def updateFitness(R, pBestPos, pBestVal, gBestPos, gBestVal):
    for i in range(len(R)):
        fit = fitFunc1(R[i])
        if fit < pBestVal[i]:
            pBestVal[i] = fit
            pBestPos[i] = R[i].copy()
            if fit < gBestVal:
                gBestVal = fit
                gBestPos[:] = R[i]
    return gBestVal

# --- メイン処理 ---
if __name__ == "__main__":
    Np, Nd, Nt = 20, 20, 1000
    c1, c2 = 2.05, 2.05
    wMin, wMax = 0.4, 0.9  # LDIWMに基づく慣性項の初期・最終値
    xMin, xMax = -5.12, 5.12
    vMin, vMax = 0.25 * xMin, 0.25 * xMax
    ITR = 10

    history = np.empty((ITR, Nt))

    for i in range(ITR):
        R = initPosition(Np, Nd, xMin, xMax)
        V = initVelocity(Np, Nd, vMin, vMax)
        pBestPos = R.copy()
        pBestVal = np.array([fitFunc1(r) for r in R])
        gBestIndex = np.argmin(pBestVal)
        gBestVal = pBestVal[gBestIndex]
        gBestPos = R[gBestIndex].copy()

        for j in range(Nt):
            w = wMax - ((wMax - wMin) / Nt) * j  # LDIWMの式
            updatePosition(R, V, xMin, xMax)
            gBestVal = updateFitness(R, pBestPos, pBestVal, gBestPos, gBestVal)
            history[i][j] = gBestVal
            updateVelocity(R, V, w, vMin, vMax, pBestPos, gBestPos, c1, c2)
    
    # フォルダ作成
    output_dir = "実験4結果"
    os.makedirs(output_dir, exist_ok=True)

    # 保存・表示
    df = pd.DataFrame(history).T
    df.plot(logy=True, xlim=[0, 1000], ylim=[1e-10, 1e6], fontsize=14, figsize=(9, 6))
    plt.xlabel("繰り返し回数", size=16)
    plt.ylabel("目的関数値", size=16)
    plt.title("LDIWM付きPSOによるRastrigin関数最適化（10回試行）", size=18)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("実験4結果/kadai4_ldiwm_graph.png")
    df.to_csv("実験4結果/kadai4_ldiwm_history.csv", index_label="世代")
    plt.show()
