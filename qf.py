import math

import numpy as np
from numpy.polynomial import polynomial, legendre
from tabulate import tabulate

from lab1 import input_value
from lab2 import print_polynomial
from lab5 import simpson


##############################################################################
# Параметры задачи
##############################################################################
a, b = 0, 1

def f(x):
    "f(x) = sin(x)"
    return math.sin(x)

def w(x):
    "w(x) = 1 / (x + 0.1)"
    return 1 / (x + 0.1)
##############################################################################


def gauss(f, a, b, N):
    s = 0
    xs, As = legendre.leggauss(N)
    for x, A in zip(xs, As):
        xm = x * (b - a) / 2 + (a + b) / 2
        s += A * f(xm)
    return s * (b - a) / 2


def gauss_mult(f, A, B, N, m):
    s = 0
    h = (B - A) / m
    for i in range(m):
        l, r = A + i * h, A + (i + 1) * h
        s += gauss(f, l, r, N)
    return s


def main():
    print(
        "Лабораторная работа №6",
        "Приближенное вычисление интегралов при помощи КФ НАСТ",
        "-----------------------------------------------------",
        sep='\n', end='\n\n')

    print(f.__doc__)
    print(w.__doc__)
    print("[a, b] = [{0}, {1}]".format(a, b))

    N = input_value("Введите N: ",
                    value_type=int, check=lambda N: N > 0)
    m = input_value("Введите число шагов m составной КФ: ",
                    value_type=int, check=lambda m: m > 0)
    print()

    J_gauss_ref = gauss_mult(lambda x: f(x) * w(x), a, b, N, m)
    print("J =", J_gauss_ref, end='\n\n')

    mu = [simpson(lambda x: w(x) * (x ** k), a, b, 100000) for k in range(2 * N)]
    print("Моменты весовой функции:", mu)

    aval = np.linalg.solve([mu[i:i+N] for i in range(N)], [-x for x in mu[-N:]])
    P = polynomial.Polynomial(aval) + polynomial.Polynomial(polynomial.polyx) ** N
    print("Ортогональный w_n(x) =", print_polynomial(P))

    xs = P.roots()
    print("Узлы КФ типа Гаусса:", xs)

    W = polynomial.Polynomial([1])
    for x_i in xs:
        W *= polynomial.Polynomial([-x_i, 1])
    W_prime = W.deriv()
    As = [gauss_mult(lambda x: w(x) * W(x) / ((x - x_i) * W_prime(x_i)), a, b, N, m) for x_i in xs]

    print("Коэффициенты КФ:", As)

    J_gauss = sum(A * f(x) for x, A in zip(xs, As))
    print("J =", J_gauss)


if __name__ == '__main__':
    main()