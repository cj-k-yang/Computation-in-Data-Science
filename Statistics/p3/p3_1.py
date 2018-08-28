import numpy as np
import matplotlib.pyplot as plt



def monte_carlo(n_iter=50000):
    monte_carlo_sum = 0
    for i in range(n_iter):
        r = np.random.rand(20)
        #print(r)
        total = 0
        for k in range(19):
            total += np.cos(r[k]/np.pi) + k*k*np.sin(r[k+1]/np.pi)
            #print(total)
        #print("____________")
        monte_carlo_sum += np.exp(total)
        if (i+1)%(n_iter/10)==0:
            print(i+1)
    return monte_carlo_sum/float(n_iter)


if __name__ == '__main__':
    integral = monte_carlo()
    print(integral)

