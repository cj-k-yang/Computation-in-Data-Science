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
    p_times = 0
    q_times = 0
    #success_time = 0

    for i in range(n_iter):
        money = 20
        while money>0 and money<200:
            #print(money)
            if random.uniform(0, 1) < p:
                money -= 1
                p_times += 1
            else:
                money += 1
                q_times += 1
        if money == 200:
            #success_time += 1
            #success += 1.0*(0.9)**180
            success += 1.0
    success = success*(0.9**180)
    success /= n_iter
    print(success)

if __name__ == '__main__':
    
    print(0.9**180)

    bet(1000)