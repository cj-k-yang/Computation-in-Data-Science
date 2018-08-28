import numpy as np
import matplotlib.pyplot as plt

  


def importance_sampling(n_iter=1000000):
    monte_carlo_sum = 0
    #g(x)~N(8,1)
    mu, sigma = 8, 1
    sample_normal = np.random.normal(mu, sigma, n_iter)
    #print(sample_normal.shape)
    #print(sample_normal)
    for i in range(n_iter):
        x = sample_normal[i]
        if x>8:
            monte_carlo_sum += np.exp(-(x**2-8*x+32))
    return monte_carlo_sum/float(n_iter)


if __name__ == '__main__':
    print(importance_sampling(1000000))    
