import sympy as sp
import numpy as np

from numpy.linalg import solve


def pmoment(p, a, b, r):
    # p - weight function
    # r - count
    # a,b - limits
    M = list(map(lambda j: sp.Integral(p * x ** j, (x, a, b)).as_sum(20, method="midpoint").n(), range(0, r)))
    M = sp.Matrix(M)
    return list(M)


def gaussqf(M, p, a, b, lf):
    l = len(M)  # count of moments
    k = l // 2  # count of equations
    M_ = []
    # Making system for a1..an

    for i in range(k - 1, 2 * k - 1):
        S = []
        for j in range(0, k):
            S.append(M[i - j])

        M_.append(S)

    m = M[k:l]

    m = [-1 * x for x in m]
    m = np.array(m, dtype='float')
    M_ = np.array(M_, dtype='float')
    x = sp.symbols('x')
    A = solve(M_, m)
    n = len(A)  # polynomial degree
    A = list(A)
    A = list(reversed(A))
    A.append(1)

    x = sp.symbols("x")
    eq = 0
    # Make equation
    for i in range(n, -1, -1):
        eq += A[i] * x ** i
    # Find nodes.
    # Nodes must be different and must be in [a,b]

    X = list(sp.solve(eq))
    X = [sp.re(x) for x in X]
    # omega and derivative of omega
    w = 1
    for i in range(len(X)):
        w = w * (x - X[i])
    dw = sp.diff(w, x)
    dw = sp.lambdify(x, dw)
    A = []
    # find coef of the quadrature formula
    for i in range(len(X)):
        A.append(sp.Integral(p * w / ((x - X[i]) * (dw(X[i]))), (x, a, b)).as_sum(20, method="midpoint").n())

    X = np.array(X, dtype=np.float)
    A = np.array(A, dtype=np.float)

    integ_sum = 0
    for i in range(len(A)):
        integ_sum = integ_sum + lf(X[i]) * A[i]
    print(integ_sum)


if __name__ == "__main__":
    x = sp.symbols("x")
    a = 1.3
    b = 2.2

    alpha = 0
    betta = 5 / 6
    p = (x - a) ** (-alpha) * (b - x) ** (-betta)

    xk = [a, (b - a) / 2, b]
    f = 4 * sp.cos(0.5 * x) * sp.exp(-5 * x / 4) + 2 * sp.sin(4.5 * x) * sp.exp(x / 8) + 2
    lf = lambda xk: 4 * np.cos(0.5 * xk) * np.exp(-5 * xk / 4) + 2 * np.sin(4.5 * xk) * np.exp(xk / 8) + 2

    M = pmoment(p, a, b, 6)
    gaussqf(M, p, a, b, lf)
    #print(sp.Integral(f * p, (x, a, b)).as_sum(20, method="midpoint").n())
