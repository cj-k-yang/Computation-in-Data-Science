import numpy as np
import matplotlib.pyplot as plt
import random
import math
# f(x) = 1/(pi*(1+x^2)) = y
# x = +- sqrt((1/(pi*y) - 1))



def bet(n_iter=1000):
    success = 0.0
    p = 18.0/38.0
    q = 20.0/38.0
    #success_time = 0
    for i in range(n_iter):
        money = 20
        while money>0 and money<200:
            #print(money)
            if random.uniform(0, 1) < p:
                money -= 1
            else:
                money += 1
        if money == 200:
            #success_time += 1
            #success += 1.0*(0.9)**180
            success += 1.0
    success = success*(0.9**180)
    success /= n_iter
    print("prob of success(old way):", success)
    
def bet_turns(n_iter=1000):
    p = 18.0/38.0
    turns = 0
    for i in range(n_iter):
        money = 20
        while money>0 and money<200:
            turns += 1
            #print(money)
            if random.uniform(0, 1) > p:
                money -= 1
            else:
                money += 1
    print("turns:", turns/n_iter)

def poisson95():
    #x = np.random.poisson(0.95, 10000)
    #plt.hist(x, 100, density=True, facecolor='g', alpha=0.75)
    #plt.show()
    return np.random.poisson(0.95)

def poisson105():
    return np.random.poisson(1.05)

def bet_poisson(n_iter=1000):
    success = 0.0
    #turns = 0
    for i in range(n_iter):
        money = 20
        to_multiply = 1.0
        while money>0 and money<200:
            #turns += 1
            money-=1
            k = poisson105()
            money+=k
            p = (0.95**k)*np.exp(-0.95)/np.math.factorial(k)
            q = (1.05**k)*np.exp(-1.05)/np.math.factorial(k)
            to_multiply *= (p/q)#*((1-p)/(1-q))
        if money == 200:
            success += 1.0*to_multiply
    success /= n_iter
    print("prob of success(new machine):", success)
    #print("turns:", turns/n_iter)

def bet_poisson_turns(n_iter=1000):
    turns = 0
    for i in range(n_iter):
        money = 20
        while money>0 and money<200:
            turns += 1
            money-=1
            k = poisson95()
            money+=k
    print("turns:", turns/n_iter)

def weight(x):
    return(((np.e**(-0.95))/(np.e**(-1.05)))*(((0.95)/(1.05))**x))

if __name__ == '__main__':
    
    #print(0.9**180)
    t = 1000
    bet(t)
    bet_turns(t)
    bet_poisson(t)
    bet_poisson_turns(t)
    """
    1. 新機器不會對玩家比較有利(贏的機率較小)
    2. 新機器與輪盤所花局數差不多，但多次平均來看新機器仍然會多花一點局數
    """