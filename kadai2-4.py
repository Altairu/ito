import os
import math
from random import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# --- Rastrigin 関数 ---
def fit_rastrigin(x):
    f = 10 * len(x)
    for xi in x:
        f += xi**2 - 10 * math.cos(2 * math.pi * xi)
    return f

# --- 粒子群共通 ---
def init_position(Np, Nd, xMin, xMax):
    return [[xMin + random()*(xMax-xMin) for _ in range(Nd)] for _ in range(Np)]

def init_velocity(Np, Nd, vMin, vMax):
    return [[vMin + random()*(vMax-vMin) for _ in range(Nd)] for _ in range(Np)]

def update_position(R, V, xMin, xMax):
    for p in range(len(R)):
        for i in range(len(R[p])):
            R[p][i] = min(max(R[p][i] + V[p][i], xMin), xMax)

def update_velocity(R, V, w, c1, c2, pBestPos, gBestPos, vMin, vMax):
    Np, Nd = len(R), len(R[0])
    for p in range(Np):
        for i in range(Nd):
            r1, r2 = random(), random()
            v = w*V[p][i] + c1*r1*(pBestPos[p][i]-R[p][i]) + c2*r2*(gBestPos[i]-R[p][i])
            V[p][i] = min(max(v, vMin), vMax)

# --- kadai2: 固定慣性項 w_const の PSO ---
def run_kadai2(Np, Nd, Nt, c1, c2, w_const, xMin, xMax, vMin, vMax, ITR):
    history = np.zeros((ITR, Nt))
    for t in range(ITR):
        R = init_position(Np, Nd, xMin, xMax)
        V = init_velocity(Np, Nd, vMin, vMax)
        M = [fit_rastrigin(r) for r in R]
        pBVal, pBPos = M[:], [r[:] for r in R]
        gBVal, gBPos = min(M), R[M.index(min(M))][:]
        for j in range(Nt):
            update_position(R, V, xMin, xMax)
            for p in range(Np):
                M[p] = fit_rastrigin(R[p])
                if M[p] < pBVal[p]:
                    pBVal[p], pBPos[p] = M[p], R[p][:]
                if M[p] < gBVal:
                    gBVal, gBPos = M[p], R[p][:]
            update_velocity(R, V, w_const, c1, c2, pBPos, gBPos, vMin, vMax)
            history[t, j] = gBVal
    return pd.DataFrame(history).mean(axis=0)

# --- kadai5: LDIWM による慣性項線形減衰 PSO （ITR 回平均化）---
def run_kadai5(Np, Nd, Nt, c1, c2, wMin, wMax, xMin, xMax, vMin, vMax, ITR):
    history = np.zeros((ITR, Nt))
    for t in range(ITR):
        R = init_position(Np, Nd, xMin, xMax)
        V = init_velocity(Np, Nd, vMin, vMax)
        M = [fit_rastrigin(r) for r in R]
        pBVal, pBPos = M[:], [r[:] for r in R]
        # 初期 gBest の値と位置を設定
        gBVal = min(M)
        gBPos = R[M.index(gBVal)][:]

        hist = []

        for j in range(Nt):
            update_position(R, V, xMin, xMax)
            for p in range(Np):
                M[p] = fit_rastrigin(R[p])
                if M[p] < pBVal[p]:
                    pBVal[p], pBPos[p] = M[p], R[p][:]
                if M[p] < gBVal:
                    gBVal, gBPos = M[p], R[p][:]
            w = wMax - ((wMax - wMin) / Nt) * j
            update_velocity(R, V, w, c1, c2, pBPos, gBPos, vMin, vMax)
            hist.append(gBVal)
        history[t, :] = hist  # hist now has length Nt

    return pd.DataFrame(history).mean(axis=0)

# --- メイン実行 ---
if __name__ == "__main__":
    # パラメータ設定
    Np, Nd, Nt = 20, 20, 1000
    c1, c2 = 2.05, 2.05
    w_const = 0.9
    wMin, wMax = 0.4, 0.9
    xMin, xMax = -5.12, 5.12
    vMin, vMax = 0.25*xMin, 0.25*xMax
    ITR = 10

    fixed_mean = run_kadai2(Np, Nd, Nt, c1, c2, w_const,
                            xMin, xMax, vMin, vMax, ITR)
    ldiwm = run_kadai5(Np, Nd, Nt, c1, c2,
                       wMin, wMax, xMin, xMax, vMin, vMax, ITR)

    # 比較 DataFrame
    df = pd.DataFrame({
        "kadai2 固定 w=0.9 平均": fixed_mean,
        "kadai5 LDIWM": ldiwm
    })
    os.makedirs("比較結果", exist_ok=True)
    df.to_csv("比較結果/kadai2_vs_kadai5.csv", index_label="世代")

    # プロット
    plt.figure(figsize=(9,6))
    x = df.index.values
    y1 = df["kadai2 固定 w=0.9 平均"].values
    y2 = df["kadai5 LDIWM"].values
    plt.plot(x, y1, label="kadai2 固定 w=0.9")
    plt.plot(x, y2, label="kadai5 LDIWM")
    plt.xlabel("世代", size=14)
    plt.ylabel("最良評価値", size=14)
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("比較結果/kadai2_vs_kadai5.png")
    plt.show()