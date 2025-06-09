import numpy as np
import math

def fitFunc1(xVals):
    return 10 * len(xVals) + sum(x**2 - 10 * math.cos(2 * math.pi * x) for x in xVals)

def initPosition(Np, Nd, xMin, xMax):
    return [[xMin + np.random.rand() * (xMax - xMin) for _ in range(Nd)] for _ in range(Np)]

def initVelocity(Np, Nd, vMin, vMax):
    return [[vMin + np.random.rand() * (vMax - vMin) for _ in range(Nd)] for _ in range(Np)]

def updateVelocity(R, V, Np, Nd, w, vMin, vMax, pBestPos, gBestPos, c1, c2):
    for p in range(Np):
        for i in range(Nd):
            r1, r2 = np.random.rand(), np.random.rand()
            V[p][i] = (
                w * V[p][i]
                + r1 * c1 * (pBestPos[p][i] - R[p][i])
                + r2 * c2 * (gBestPos[i] - R[p][i])
            )
            V[p][i] = np.clip(V[p][i], vMin, vMax)

def updatePosition(R, V, Np, Nd, xMin, xMax):
    for p in range(Np):
        for i in range(Nd):
            R[p][i] += V[p][i]
            R[p][i] = np.clip(R[p][i], xMin, xMax)

def updateFitness(R, Np, pBestPos, pBestVal, gBestPos, gBestValue):
    for p in range(Np):
        fit = fitFunc1(R[p])
        if fit < gBestValue:
            gBestValue = fit
            gBestPos[:] = R[p][:]
        if fit < pBestVal[p]:
            pBestVal[p] = fit
            pBestPos[p][:] = R[p][:]
    return gBestValue

if __name__ == "__main__":
    Np, Nd, Nt = 20, 20, 1000
    c1, c2 = 2.05, 2.05
    wMin, wMax = 0.4, 0.9  # LDIWM用
    xMin, xMax = -5.12, 5.12
    vMin, vMax = 0.25 * xMin, 0.25 * xMax
    gBestValue = float("inf")
    pBestValue = [float("inf")] * Np
    pBestPos = [ [0]*Nd for _ in range(Np) ]
    gBestPos = [0] * Nd
    history = []

    R = initPosition(Np, Nd, xMin, xMax)
    V = initVelocity(Np, Nd, vMin, vMax)
    for j in range(Nt):
        updatePosition(R, V, Np, Nd, xMin, xMax)
        gBestValue = updateFitness(R, Np, pBestPos, pBestValue, gBestPos, gBestValue)
        history.append(gBestValue)
        w = wMax - ((wMax - wMin) / Nt) * j  # LDIWM 慣性項
        updateVelocity(R, V, Np, Nd, w, vMin, vMax, pBestPos, gBestPos, c1, c2)

    # グラフ描画等は課題2のコードを参考にしてください