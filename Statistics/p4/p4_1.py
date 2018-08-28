import numpy as np
import matplotlib.pyplot as plt
import random
import math
# f(x) = 1/(pi*(1+x^2)) = y
# x = +- sqrt((1/(pi*y) - 1))

def func(x):
    return x**2/np.sqrt(2*np.pi)*np.exp(-x**2/2)

def plot():
    xl = np.linspace(-10,10,10000)
    n = np.random.normal(0,1,100000)
    n2 = np.random.normal(0,np.sqrt(2),100000)
    plt.hist(n, 100, normed=1, facecolor='b', alpha=0.2)
    plt.hist(n2, 100, normed=1, facecolor='g', alpha=0.2)
    plt.plot(xl, [func(x) for x in xl], color='r')
    plt.show()


if __name__ == '__main__':
    plot()