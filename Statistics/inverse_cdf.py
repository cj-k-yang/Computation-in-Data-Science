import numpy as np
import matplotlib.pyplot as plt
import random
import math
# f(x) = 1/(pi*(1+x^2)) = y
# x = +- sqrt((1/(pi*y) - 1))

PI = math.pi
lambd = 0.5

def get_next(x, y):
    rand = random.uniform(0, 1)
    y.append(rand)
    x.append(-math.log(1-rand)/lambd)                

print(math.pi, PI)

x = []
y = []
x.append(0)
y.append(0)
for i in range(10000):
    get_next(x, y)

x = np.array(x)
y = np.array(y)
print(x.shape, y.shape)

plt.hist(x, 500, density=True, facecolor='g', alpha=0.75)
plt.axis([0, 10, 0, 1])
plt.grid(True)
plt.show()
