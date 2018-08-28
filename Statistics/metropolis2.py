import random
import collections


def metrosamp(ITER):
    P = {0: 0.2, 1: 0.8}
    Q = {1: 0, 0: 1}
    X = []
    xt = 0
    for i in range(ITER):
        xtp1 = Q[xt]
        alpha = min(1., P[xtp1] / P[xt])
        if random.random() <= alpha:
            xt = xtp1
        X.append(xt)
    return X


def count(X):
    counter = collections.Counter(X)
    for key in counter:
        print(key, counter[key]) 

if __name__ == "__main__":
    X = metrosamp(10000)
    count(X)
    #print(X)