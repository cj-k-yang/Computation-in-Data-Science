import numpy as np
from collections import Counter

def gibbssamp(m):
    P = np.matrix([[0.5, 0.2], [0.1, 0.2]])
    condi = [np.divide(P, np.sum(P, axis=0)), np.divide(P, np.sum(P, axis=1)).T]
    x = [0, 0]
    samples = []
    for i in range(m):
        for j in range(2):
            one_prob = condi[j][1, x[(j + 1) % 2]]
            x[j] = np.random.binomial(1, one_prob, 1)[0]
            samples.append([x[0], x[1]])
    return samples


def count(samples):
    c = Counter()
    for s in samples:
        c.update(["(%s,%s)" % (s[0], s[1])])
    for k in c.keys():
        print(k, c[k])

if __name__ == "__main__":
    X = gibbssamp(100000)
    count(X)
    #print(X)