import numpy as np
import random
import collections
import matplotlib.pyplot as plt
import math
import scipy.stats

def p_func_raw(x, a, b):
    S1 = ((b ** a) / math.gamma(a))
    S2 = x ** (a - 1)
    S3 = math.exp(-b * x)
    return S1 * S2 * S3  # * S4


def p_func(y):
    return p_func_raw(y, 2, 1)

def q_func(beta):
    return np.random.exponential(beta)

def q_func_pdf(x, beta):
    return scipy.stats.expon.pdf(x, scale=beta)

def metrohast(M):
    X = []
    beta = 5.
    xt = beta
    for i in range(M):
        aj = q_func(beta)
        c = q_func_pdf(xt, beta) / q_func_pdf(aj, beta)
        alpha = min(1., (p_func(aj) / p_func(xt)) * c)
        if random.random() <= alpha:
            xt = aj
        X.append(xt)
    return X

def draw(S):
    n, bins, patches = plt.hist(S, 100, normed=1, facecolor='b', alpha=0.2)
    plt.plot(bins, [p_func(x) for x in bins], color='r')
    plt.show()

if __name__ == "__main__":
    X = metrohast(1000)
    draw(X)
    print(X[:100])