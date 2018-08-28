import numpy as np
from numpy.linalg import svd, qr, matrix_rank
from PIL import Image
import matplotlib.pyplot as plt

def rsvd(A, k):
    m = A.shape[0]
    n = A.shape[1]
    Omega = np.random.randn(n,k)
    Y = A@Omega
    Q, R = qr(Y)
    print(Q.shape)
    print(R.shape)
    B = Q.T@A
    U_hat, Sigma, Vt = svd(B)
    print(U_hat.shape, Sigma.shape, Vt.shape)
    U = Q@U_hat
    #V = 1
    #print(Sigma)

    """
    Psi = np.random.randn(m,k)
    X = A.T@Psi
    Qx, Rx = qr(X)
    print(Qx.shape)
    print(Rx.shape)
    B2 = Qx.T@A.T
    U_hat, Sigma, V_hat = svd(B2)
    V = Qx@U_hat
    
    print(Sigma)"""
    """
    Psi = np.random.randn(m,k)
    Y = A.T@Psi
    Q, R = qr(Y)
    B = Q.T@A.T
    U_hat, Sigma, V_hat = svd(B)
    
    V = Q@V_hat.T
    """

    return U, Sigma, Vt

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

if __name__ == '__main__':
    pic = np.array(Image.open('Best-Black-and-White-Wallpapers-for-Walls.jpg'))
    print(pic.shape)
    #plt.figure()
    #plt.imshow(pic, cmap='gray', aspect="equal", interpolation="none")
    #plt.show()
    pic = rgb2gray(pic)
    print(pic.shape)
    
    #plt.figure()
    #plt.imshow(pic, cmap='gray', aspect="equal", interpolation="none")
    #plt.show()
    #u, sigma, vt = svd(pic)
    #print(sigma)
    #print(matrix_rank(pic))
    k = 1000
    ur, sigmar, vtr = rsvd(pic, k)

    print(ur.shape, sigmar.shape, vtr.shape)
    print(vtr)
    #print(sigmar)
    plt.figure()
    plt.imshow(ur@np.diag(sigmar)@vtr[:k,:], cmap='gray', aspect="equal", interpolation="none")
    plt.show()