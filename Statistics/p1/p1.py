import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GMM



if __name__ == "__main__":
    all_weight = []
    with open('data1.txt', 'r') as f:
        for line in f.readlines():
            #print(line)
            each = line
            each = each.split()
            for e in each:
                all_weight.append(float(e))


    all_weight = np.array(all_weight).reshape(-1,1)
    print(all_weight.shape)

    plt.figure()
    plt.hist(all_weight, 50, density=True, facecolor='g', alpha=0.75)
    plt.show()

    """
    ############################ KMEAN ############################

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(all_weight)
    p = kmeans.predict(all_weight)
    print(p)
    print(kmeans)
    p = p.reshape(-1,1)

    f = np.where(p==1, all_weight, 0)
    index = np.argwhere(f==0)
    f = np.delete(f, index)

    m = np.where(p==0, all_weight, 0)
    index = np.argwhere(m==0)
    m = np.delete(m, index)

    print(f.shape, m.shape)
    plt.figure()
    plt.hist(f, 75, density=False, facecolor='g', alpha=0.75)
    #plt.show()

    #plt.figure()
    plt.hist(m, 150, density=False, facecolor='b', alpha=0.75)
    plt.show()

    ###############################################################
    """
    ############################# GMM #############################

    gmm = GMM(n_components=2).fit(all_weight)
    p = gmm.predict(all_weight)
    
    p = p.reshape(-1,1)

    f = np.where(p==1, all_weight, 0)
    index = np.argwhere(f==0)
    f = np.delete(f, index)

    m = np.where(p==0, all_weight, 0)
    index = np.argwhere(m==0)
    m = np.delete(m, index)

    print(f.shape, m.shape)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[8, 5])
    #plt.figure()
    ax.hist(f, bins=30, density=False, facecolor='g', alpha=0.75)
    #plt.show()

    #plt.figure()
    ax.hist(m, bins=60, density=False, facecolor='b', alpha=0.75)
    plt.show()
    ###############################################################
