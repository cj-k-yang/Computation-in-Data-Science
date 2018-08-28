import numpy as np
import matplotlib.pyplot as plt
import random
import math
# f(x) = 1/(pi*(1+x^2)) = y
# x = +- sqrt((1/(pi*y) - 1))

PI = math.pi

def get_next(x, y, direc): #dir: true--> next point is y dir
    if direc==0:
        y.append(random.uniform(0, 1/(PI*(1+x[-1]**2))))
        x.append(x[-1])                
    else:
        lim = math.sqrt((1/(PI*y[-1]) - 1))
        x.append(random.uniform(-lim, lim))
        y.append(y[-1])


x = []
y = []
x.append(0)
y.append(0)
for i in range(10000):
    direc = i%2
    get_next(x, y, direc)

x = np.array(x)
y = np.array(y)
print(x.shape, y.shape)


def draw(S):
    n, bins, patches = plt.hist(S, 10000, normed=1, facecolor='b', alpha=0.2)
draw(x)
plt.scatter(x,y)
plt.xlim(-10, 10)
plt.ylim(0, np.max(y)+0.1)
plt.show()