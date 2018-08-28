import numpy as np

a = np.arange(3*6).reshape(3,6)

b = np.arange(3*6).reshape(3,6)

c = np.arange(5*5).reshape(5,5)
d = np.array([0,1,0,0,1])


print(a)
print(a*b)

print(np.sum(a*b))

print(c*d)