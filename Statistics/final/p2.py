import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.special import gamma
import collections 
import scipy.stats

def a_func(x):
    return np.sin(np.sqrt(x))*np.exp(-100*x)
def a_func_change_var(x):
    return np.sin(np.sqrt(1/x-1))*np.exp(-100*(1/x-1))*(1/x)*(1/x)




def b_func(x):
    return np.log(1.0+x)*np.exp(-98*(x**2.01))
def b_change_var(x):
    return np.log(1/x)*np.exp(-98*((1/x-1)**2.01))*(1/x)*(1/x)

def plot_a():
    xl = np.linspace(0,10,10000)
    plt.plot(xl, [a_func(x) for x in xl], color='r')
    plt.axis([0, 0.15, 0, 0.1])
    plt.show()

def a(n_iter=500000):
    monte_carlo_sum = 0
    for i in range(n_iter):
        r = random.uniform(0, 1)
        monte_carlo_sum += a_func_change_var(r)
    return monte_carlo_sum/float(n_iter)

def plot_b():
    xl = np.linspace(0,10,10000)
    plt.plot(xl, [b_func(x) for x in xl], color='r')
    plt.axis([0, 0.5, 0, 0.1])
    plt.show()

def b(n_iter=500000):
    monte_carlo_sum = 0
    for i in range(n_iter):
        r = random.uniform(0, 1)
        monte_carlo_sum += b_change_var(r)
    return monte_carlo_sum/float(n_iter)

#change variable
def c(n_iter=100000):
    monte_carlo_sum = 0
    for i in range(n_iter):
        r = np.random.rand(20)
        exp = []
        total = 1.0
        for k in range(19):
            x1 = r[k]
            x2 = r[k+1]
            #print(x1, x2)
            #total += (np.cos(1/x1-1) + (k+1)*(k+1)*np.log(1/x2))*((1/x1-1)**2+(1/x2-1)**2)/(k+1)
            total *= (np.cos(1.0/x1-1) + (k+1)*(k+1)*np.log(1.0/x2))*np.exp(-((1.0/x1-1)**2+(1.0/x2-1)**2)/(k+1))
            #print(total)
            #print(total)
        for k in range(20):
            x = r[k]
            total*=(1.0/x)*(1.0/x)
        #print("____________")
        monte_carlo_sum += total
    return monte_carlo_sum/float(n_iter)




if __name__ == '__main__':
    #iter = 1000
    #integral = monte_carlo()
    #print(integral)
    """變數變換"""
    ans_a = a()
    print("a:", ans_a)
    #plot_a()

    ans_b = b()
    print("b:", ans_b)
    #plot_b()
    
    ans_c = c()
    print("c:", ans_c)

    """
    a. , b.  皆使用變數變換使積分範圍變為 0~1，取 u = 1/(1+x)
    變數變換完後則取 uniform(0,1) 帶入換成 u 後的式子

    c. 小題也用一樣的方法將 x1~x20 變換為 u1~u20
    每次 iteration 先取好20點 uniform(0,1) 然後依序帶入 u1~u20
    """

