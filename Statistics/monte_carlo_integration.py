import numpy as np
import matplotlib.pyplot as plt
import random
import math

# target function: y = x^2


N = 10000

def mc_integral1(N):  # WRONG
    total = 0
    xn = []
    yn = []
    for i in range(N):
        x = random.uniform(0,1)
        y = random.uniform(0,1)
        if y<x*x:
            total += 1
            xn.append(x)
            yn.append(y)
    
    return np.array(xn), np.array(yn), total/N
    
def mc_integral2(N):
    f = lambda x: x**2
    xlow = 0.
    xhigh = 1.

    # randomly distributed rectangle mid-points
    xvalues = xlow + (xhigh-xlow)*np.random.random(N) 
    fvalues = f(xvalues) # f(x_i) for each rectangle
    areas = fvalues * (xhigh-xlow)/N # Area for each rectangle

    return sum(areas)
    

if __name__ == "__main__":
    x1, y1, a1 = mc_integral1(N)
    a2 = mc_integral2(N)
    
    print("Integral: ", a1, a2)
    print("Error: ", abs(1.0/3.0 - a1), abs(1.0/3.0 - a2))
    plt.scatter(x1,y1)
    plt.show()