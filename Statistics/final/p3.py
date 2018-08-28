import random
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import skimage.io
import scipy.io

class IsingGrid:
    def __init__(self, height, width, extfield, invtemp):
        self.width, self.height, self.extfield, self.invtemp = height, width, extfield, invtemp
        self.grid = np.zeros([self.width, self.height], dtype=np.int8) + 1
        
    def plot(self):
        plt.imshow(self.grid, cmap='gray', aspect="equal", interpolation="none", vmin=-1, vmax=1)
    
    def neighbours(self, x, y):
        n = []
        if x == 0:
            n.append( (self.width-1, y) )
        else:
            n.append( (x-1, y) )
        if x == self.width-1:
            n.append( (0, y) )
        else:
            n.append( (x+1, y) )
        if y == 0:
            n.append( (x, self.height-1) )
        else:
            n.append( (x, y-1) )
        if y == self.height-1:
            n.append( (x, 0) )
        else:
            n.append( (x, y+1) )
        return n
    
    def local_energy(self, x, y):
        return self.extfield + sum( self.grid[xx,yy] for (xx, yy) in self.neighbours(x, y) )

    def total_energy(self):
        energy = - self.extfield * np.sum(self.grid)
        energy += - sum( self.grid[x, y] * sum( self.grid[xx, yy] for (xx, yy) in self.neighbours(x, y) )
                        for x in range(self.width) for y in range(self.height) ) / 2
        print(energy)
        return energy
    
    def probability(self):
        print(- self.invtemp * self.total_energy())
        return np.exp( - self.invtemp * self.total_energy() )

    def gibbs_move(self):
        n = np.random.randint(0, self.width * self.height)
        y = n // self.width
        x = n % self.width
        p = 1 / (1 + np.exp(-2 * self.invtemp * self.local_energy(x,y)))
        if np.random.random() <= p:
            self.grid[x,y] = 1
        else:
            self.grid[x,y] = -1


class IsingGridVaryingField(IsingGrid):
    def __init__(self, height, width, extfield, invtemp):
        super().__init__(height, width, 0, invtemp)
        self.vextfield = extfield
        #print(self.vextfield)
        
    def local_energy(self, x, y):
        return self.vextfield[x,y] + sum( self.grid[xx,yy] for (xx, yy) in self.neighbours(x, y) )
        
def IsingDeNoise(noisy, q, T, burnin = 100000, loops = 300000):
    h = 0.5 * np.log(q / (1-q))

    beta = 1.0/T
    gg = IsingGridVaryingField(noisy.shape[0], noisy.shape[1], h*noisy, beta)
    gg.grid = np.array(noisy)
    
    # Burn-in
    for _ in range(burnin):
        gg.gibbs_move()
    
    # Sample
    avg = np.zeros_like(noisy).astype(np.float64)
    for _ in range(loops):
        gg.gibbs_move()
        avg += gg.grid
        #if (_+1)%50000==0:
            #print((_+1)/50000)
            #axes[int((_+1)/50000)].imshow(gg.grid, cmap='gray', aspect="equal", interpolation="none", vmin=-1, vmax=1)
    return avg / loops, gg.grid

def IsingProb(noisy, q, T, burnin = 100000, loops = 300000):
    h = 0.5 * np.log(q / (1-q))
    beta = 1.0/T
    gg = IsingGridVaryingField(noisy.shape[0], noisy.shape[1], h*noisy, beta)
    gg.grid = np.array(noisy)
    return gg.probability()

if __name__ == "__main__":
    noisy_pic = scipy.io.loadmat("data4")['imgMat']
    
    text = []
    with open("data4.txt") as f:
        for line in f.readlines():
            for i in line.split():
                text.append(int(i))

    #print(len(text))
    noisy_pic2 = np.array(text).reshape(100,100).T

    """
    Ts = [0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 5.0, 10.0]
    for i in range(8):
        print("______________________")
        print(i)
        print(IsingProb(noisy_pic, 0.625, Ts[i]))
        print("______________________")
        #print(np.exp(1424))
    print()
    """
    fig, axes = plt.subplots(ncols=9, figsize=(11,6))
    avgs = []

    Ts = [0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 5.0, 10.0]
    for i in range(8):
        print(i)
        avg, last = IsingDeNoise(noisy_pic, 0.625, Ts[4], 100000, 300000)
        avg[avg >= 0] = 1
        avg[avg < 0] = -1
        avgs.append(avg.astype(np.int))
        #axes[i+1].imshow(avg.astype(np.int), cmap='gray', aspect="equal", interpolation="none", vmin=-1, vmax=1)
        axes[i+1].imshow(last.astype(np.int), cmap='gray', aspect="equal", interpolation="none", vmin=-1, vmax=1)
        

    #axes[6].imshow(sum(avgs), cmap='gray', aspect="equal", interpolation="none", vmin=-1, vmax=1)
    axes[0].imshow(noisy_pic, cmap='gray', aspect="equal", interpolation="none", vmin=-1, vmax=1)
    plt.show()
    """
    用 gibbs sampling 變換 ising model 中的值
    分別用不同 T 測試 (0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 5.0, 10.0)
    所畫圖第一章為原圖，由左至右為依序取上述 T 進行 gibbs sampling 的結果 (共9張圖) 
    跑了多次結果看來效果最好的為 T = 3.0 , 5.0
    """
    