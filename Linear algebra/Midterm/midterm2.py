import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.sparse.linalg as linalg

np.set_printoptions(precision=15)
def create_blur_matrix(n):
    B = np.zeros((n,n))
    B[0,0] = 1/float(2)
    B[0,1] = 1/float(4)
    B[-1,-1] = 1/float(2)
    B[-1, -2] = 1/float(5)
    if n>2:
        for i in range(1,n-1):
            B[i,i-1] = 1/float(5)
            B[i,i] = 1/float(2)
            B[i,i+1] = 1/float(4)
    return B

def blur_operation(image, kv, kh):
    m = image.shape[0]
    n = image.shape[1]
    B = create_blur_matrix(m)
    C = create_blur_matrix(n)
    #C = np.transpose(C)
    blurImage = image
    for i in range(kv):
        blurImage = B@blurImage
    for i in range(kh):
        blurImage = blurImage@C
    return blurImage

print(create_blur_matrix(5))   

madrill = np.array(Image.open("./bs_madrill.jpg")) 
blurred = blur_operation(madrill, 8, 15)
print(blurred)


plt.figure()
plt.axis('off')
plt.imshow(blurred, cmap='gray')
plt.show()
plt.imsave('./bs_madrill_blur.jpg',blurred,cmap='gray')

def deblur_operation(blurimage, *args):
    
    