import math

# Sphere関数（単峰性）
def fitFuncSphere(xVals):
    """Sphere function"""
    return sum([x**2 for x in xVals])

# Rosenbrock関数（変数依存型、多峰性）
def fitFuncRosenbrock(xVals):
    """Rosenbrock (Star) function"""
    D = len(xVals)
    return sum([100 * (xVals[i] - xVals[i-1])**2 + (xVals[i] - 1.0)**2 for i in range(1, D)])

# Griewank関数（多峰性、変数間の依存有）
def fitFuncGriewank(xVals):
    """Griewank function"""
    sum_term = sum([x**2 for x in xVals]) / 4000
    prod_term = 1
    for i in range(len(xVals)):
        prod_term *= math.cos(xVals[i] / math.sqrt(i+1))
    return sum_term - prod_term + 1

if __name__ == "__main__":
    # テスト例
    x = [0, 0, 0]
    print("Sphere:", fitFuncSphere(x))
    print("Rosenbrock:", fitFuncRosenbrock(x))
    print("Griewank:", fitFuncGriewank(x))