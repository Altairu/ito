import os
import math
import random
import matplotlib.pyplot as plt
import pandas as pd

# 日本語フォント設定（Noto CJKなどを使う）
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

# === ベンチマーク関数 ===
def fitFuncSphere(xVals):
    return sum(x**2 for x in xVals)

def fitFuncRosenbrock(xVals):
    return sum(100 * (xVals[i] - xVals[i-1])**2 + (xVals[i-1] - 1)**2 for i in range(1, len(xVals)))

def fitFuncGriewank(xVals):
    sum_term = sum(x**2 for x in xVals) / 4000
    prod_term = 1
    for i in range(len(xVals)):
        prod_term *= math.cos(xVals[i] / math.sqrt(i + 1))
    return sum_term - prod_term + 1

def fitFuncRastrigin(xVals):
    return 10 * len(xVals) + sum(x**2 - 10 * math.cos(2 * math.pi * x) for x in xVals)

# === 初期化関数 ===
def initPosition(Np, Nd, xMin, xMax):
    return [[xMin + random.random() * (xMax - xMin) for _ in range(Nd)] for _ in range(Np)]

def initVelocity(Np, Nd, vMin, vMax):
    return [[vMin + random.random() * (vMax - vMin) for _ in range(Nd)] for _ in range(Np)]

# === 更新関数 ===
def updateVelocity(R, V, Np, Nd, j, w, vMin, vMax, pBestPos, gBestPos, c1, c2):
    for p in range(Np):
        for i in range(Nd):
            r1 = random.random()
            r2 = random.random()
            V[p][i] = (w * V[p][i] +
                       c1 * r1 * (pBestPos[p][i] - R[p][i]) +
                       c2 * r2 * (gBestPos[i] - R[p][i]))
            V[p][i] = max(min(V[p][i], vMax), vMin)

def updatePosition(R, V, Np, Nd, xMin, xMax):
    for p in range(Np):
        for i in range(Nd):
            R[p][i] += V[p][i]
            R[p][i] = max(min(R[p][i], xMax), xMin)

def updateFitness(R, M, Np, pBestPos, pBestVal, gBestPos, gBestValue, fitFunc):
    for p in range(Np):
        M[p] = fitFunc(R[p])
        if M[p] < gBestValue:
            gBestValue = M[p]
            gBestPos[:] = R[p][:]
        if M[p] < pBestVal[p]:
            pBestVal[p] = M[p]
            pBestPos[p][:] = R[p][:]
    return gBestValue

# === PSO実行関数 ===
def runPSO(fitFunc, name, Np=30, Nd=30, Nt=500,
           xMin=-5.12, xMax=5.12,
           vMin=None, vMax=None,
           wMin=0.4, wMax=0.9, c1=2.05, c2=2.05):
    
    vMin = vMin if vMin is not None else 0.25 * xMin
    vMax = vMax if vMax is not None else 0.25 * xMax

    R = initPosition(Np, Nd, xMin, xMax)
    V = initVelocity(Np, Nd, vMin, vMax)
    M = [fitFunc(R[p]) for p in range(Np)]
    pBestVal = M[:]
    pBestPos = [r[:] for r in R]
    gBestValue = min(M)
    gBestPos = R[M.index(gBestValue)][:]
    history = [gBestValue]

    for j in range(Nt):
        w = wMax - (wMax - wMin) * j / Nt
        updatePosition(R, V, Np, Nd, xMin, xMax)
        gBestValue = updateFitness(R, M, Np, pBestPos, pBestVal, gBestPos, gBestValue, fitFunc)
        updateVelocity(R, V, Np, Nd, j, w, vMin, vMax, pBestPos, gBestPos, c1, c2)
        history.append(gBestValue)
    
    return name, history

# === 実験実行 ===
def main():
    output_dir = "./実験1結果"
    os.makedirs(output_dir, exist_ok=True)

    benchmark_funcs = [
        (fitFuncSphere, "Sphere関数"),
        (fitFuncRosenbrock, "Rosenbrock関数"),
        (fitFuncGriewank, "Griewank関数"),
        (fitFuncRastrigin, "Rastrigin関数")
    ]

    all_histories = []
    df_data = {}

    for func, name in benchmark_funcs:
        print(f"{name} のPSO最適化を実行中...")
        name, history = runPSO(func, name)
        all_histories.append((name, history))
        df_data[name] = history

    df = pd.DataFrame(df_data)
    df.index.name = "世代"
    df.to_csv(os.path.join(output_dir, "収束履歴表.csv"))

    plt.figure(figsize=(10, 6))
    for name, history in all_histories:
        plt.plot(history, label=name)
    plt.xlabel("世代")
    plt.ylabel("最良評価値")
    plt.title("PSOによるベンチマーク関数の収束過程")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "収束グラフ.png"))
    plt.show()

if __name__ == "__main__":
    main()
