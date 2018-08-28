import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.special import gamma
import collections 
import scipy.stats


def get_next_a(x, y):
    rand = random.uniform(0, 1)
    y.append(rand)
    #x.append(-math.log(1-rand)/lambd)
    if rand<0.5:
        x.append(math.log(2.0*rand))
        #print(math.log(2.0*rand))
    else:
        x.append(-math.log(2.0*(1.0-rand)))

def a():
    x = []
    y = []
    #x.append(0)
    #y.append(0)
    for i in range(1000):
        get_next_a(x, y)

    x = np.array(x)
    y = np.array(y)
    #print(x.shape, y.shape)

    axes[0].hist(x, 100, density=True, facecolor='g', alpha=0.75)
    #plt.axis([-10, 10, 0, 1])
    #plt.grid(True)
    #plt.show()


def laplace(x):
    return 0.5*np.exp(-np.abs(x))


def student_t(x):
    return gamma(2)/(np.sqrt(3*np.pi)*gamma(3.0/2.0))*(1+x*x/3.0)**-2

def test_cover():
    xl = np.linspace(-10,10,10000)
    #n2 = np.random.standard_t(3,100000)
    #plt.hist(n2, 1000, normed=1, facecolor='b', alpha=0.2)
    plt.plot(xl, [laplace(x) for x in xl], color='r')
    plt.plot(xl, [2*student_t(x) for x in xl], color='b')
    plt.show()

def b_reject_sampling():
    wanted = []
    cg = np.random.standard_t(3,100000)
    """for i in cg:
        to_reject = laplace(i)/(2.0*student_t(i))
        rand = random.uniform(0, 1)
        if rand<to_reject:
            wanted.append(i)"""
    
    while len(wanted)<1000:
        cg = np.random.standard_t(3)
        to_reject = laplace(cg)/(2.0*student_t(cg))
        rand = random.uniform(0, 1)
        if rand<to_reject:
            wanted.append(cg)

    wanted = np.array(wanted)
    #print(wanted.shape)
    #plt.hist(cg, 1000, normed=1, facecolor='g', alpha=0.2)
    axes[1].hist(wanted, 100, normed=1, facecolor='b', alpha=0.75)
    plt.axis([-10, 10, 0, 1])
    #plt.show()
        


def q_func_pdf(x):
    return scipy.stats.norm(0, 100).pdf(x)


def metrohast(n_iter=1000):
    X = []
    xt = 0
    for i in range(n_iter):
        y = np.random.normal(0,100)
        #print(aj)
        c = q_func_pdf(xt) / q_func_pdf(y)
        #print((laplace(y) / laplace(xt)) * c)
        alpha = min(1., (laplace(y) / laplace(xt)) * c)
        #print(alpha)
        if random.random() <= alpha:
            xt = y
        X.append(xt)
    return X



def c():
    wanted = metrohast(1000)
    wanted = np.array(wanted)
    axes[2].hist(np.random.normal(0,100,10000), 100, normed=1, facecolor='g', alpha=0.75)
    axes[2].hist(wanted, 100, density=True, facecolor='b', alpha=0.75)

def laplace_inv(y):
    return -math.log(2.0*y)

def get_next_d(x, y, direc): #dir: 0--> next point is y dir
    if direc==0:
        y.append(random.uniform(0, laplace(x[-1])))
        x.append(x[-1])                
    else:
        lim = laplace_inv(y[-1])
        x.append(random.uniform(-lim, lim))
        y.append(y[-1])

def d():
    x = []
    y = []
    x.append(0)
    y.append(0.5)
    for i in range(999):
        direc = i%2
        get_next_d(x, y, direc)
    x = np.array(x)
    y = np.array(y)
    axes[3].hist(x, 100, normed=1, facecolor='b', alpha=0.75)
    #plt.axis([-10, 10, 0, 1])

if __name__=='__main__':
    
    fig, axes = plt.subplots(ncols=4, figsize=(11,6))
    a()

    #test_cover()
    #take c = 2
    b_reject_sampling()
    c()
    d()
    xl = np.linspace(-10,10,10000)
    for i in range(4):
        axes[i].plot(xl, [laplace(x) for x in xl], color='r')
        axes[i].axis([-10, 10, 0, 0.6])
    plt.show()
    """
    a. 取 uniform(0,1) 帶入 CDF 的 inverse 來取樣
    
    b. 先用 test_cover() 來看兩個函數的覆蓋情況，然後選用c=2來做 reject sampling 取樣
    
    c. 實作 metropolis hasting，在取下一步的過程中如果 uniform(0,1) 小於 acceptance probability 則停留在同一步，
       在圖中有畫出 normal(0,100) 發現其下一步被接受的機率不高，故 1000 個樣本中會有許多重複值
    
    d. 輪流替換 x,y 軸實作 slice sampling，其取得樣本與初始值有關

    e. 圖由左到右依序為a, b, c, d
       以效率來看 a 的效果最好，因為不會多算不必要的值。
       b 需要多算被reject的點，
       而c會有可能停留在同一點而浪費，且以N(x,100)會造成接受新值的機率過低，在圖形上來看N(x,100)並不是一個好的proposal
       d則是要多去取不同軸的值，並且在圖形上看起來較不好
    """