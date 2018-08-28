import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import operator
import copy
import sys

np.random.seed(8787)
#print(sklearn.preprocessing.normalize(np.array([np.sum(A_head), A-np.sum(A_head)]).reshape(-1,1), norm='l1', axis=0).ravel())

def info():
    coinAB = []
    head_tail = []
    all_t = []

    with open('data_2.txt', 'r') as f:
        for line in f.readlines():
            #print(line)
            each = line
            each = each.split()
            for e in each:
                all_t.append(float(e))

    #print(all_t)
    head_tail = all_t[:1000]
    coinAB = all_t[1000:]

    head_tail = np.array(head_tail)
    coinAB = np.array(coinAB)
    head_count = np.sum(np.where(head_tail==1, 1, 0))
    tail_count = 1000 - head_count
    print("total head:", head_count)
    print("total tail:", tail_count)

    head = np.where(head_tail==1, 1, 0)
    tail = np.where(head_tail==2, 1, 0)

    A = np.where(coinAB==1, 1, 0)
    B = np.where(coinAB==2, 1, 0)

    A_count = np.sum(np.where(coinAB==1, 1, 0))
    B_count = np.sum(np.where(coinAB==2, 1, 0))

    A_head = np.where(coinAB==1, head, 0)
    B_head = np.where(coinAB==2, head, 0)

    A_head_count = np.sum(A_head)
    A_tail_count = A_count - A_head_count

    B_head_count = np.sum(B_head)
    B_tail_count = B_count - B_head_count

    print("Coin A count:", A_count)
    print("head", A_head_count, "tail", A_tail_count)
    print("head", A_head_count/A_count, "tail", A_tail_count/A_count)
    print("Coin B count:", B_count)
    print("head", B_head_count, "tail", B_tail_count)
    print("head", B_head_count/B_count, "tail", B_tail_count/B_count)

    a = turnObservationToSequence(coinAB-1)
    b = count_seq(a)
    #print(a)
    print("trans_state:", b)
    print("0->1:", b['01']/(b['01']+b['00']))
    print("1->0:", b['10']/(b['10']+b['11']))
    return head_tail

def turnObservationToSequence(observation):
    sequence = []
    for i in range(observation.shape[0]-1):
        sequence.append(str(int(observation[i]))+str(int(observation[i+1])))
    return sequence

def count_seq(sequence):
    count = {}
    for i in sequence:
        if i not in count:
            count[i] = 1
        else:
            count[i] += 1
    return count

def initTransition(n_states):
    pi = np.random.random((n_states,n_states))
    row_sums = pi.sum(axis=1)
    pi = pi / row_sums[:, np.newaxis]
    return pi

def initEmission(n_states, n_emissions):
    pi = np.random.random((n_states,n_emissions))
    row_sums = pi.sum(axis=1)
    #print(row_sums)
    pi = pi / row_sums[:, np.newaxis]
    return pi

def initPi(n_states):
    pi = np.random.random((n_states)).reshape(-1,1)
    row_sums = pi.sum(axis=0)
    #print(row_sums)
    pi = pi / row_sums[:, np.newaxis]
    return pi    

def forward(A, B, pi, seq):
    T = len(seq)
    n_states = pi.shape[0]
    alpha = []
    for i in range(n_states):
        alpha.append([])
        to_append = pi[i]*B[i,seq[0]]
        alpha[i].append(to_append[0])

    """for k in range(1, T):
        np_alpha = np.array([np.array(xi) for xi in alpha])
        sum_alpha = np.sum(np_alpha, axis=1)
        #print("SA")
        #print(sum_alpha)
        for i in range(n_states):
            sum_alpha_aji = 0.0
            for j in range(n_states):
                sum_alpha_aji += sum_alpha[j]*A[j,i]
            #print(sum_alpha_aji)
            to_append = B[i,seq[0]]*sum_alpha_aji
            #print(to_append)
            alpha[i].append(to_append)
    """
    for k in range(1, T):
        for i in range(n_states):
            sum_alpha_aji = 0.0
            for j in range(n_states):
                sum_alpha_aji += alpha[j][k-1]*A[j,i]
            to_append = B[i, seq[k]]*sum_alpha_aji
            alpha[i].append(to_append)

    alpha = np.array([np.array(xi) for xi in alpha])
    #print(alpha)
    #print(alpha.shape)
    return alpha

def backward(A, B, pi, seq):
    T = len(seq)
    n_states = pi.shape[0]
    beta = []
    for i in range(n_states):
        beta.append([])
        beta[i].append(1)
    
    for k in range(1, T):
        for i in range(n_states):
            sum_beta_aji = 0.0
            for j in range(n_states):
                sum_beta_aji += beta[j][k-1]*A[i,j]*B[j, seq[T-k]]
            to_append = sum_beta_aji
            beta[i].append(to_append)

    
    #np_beta = np.array([np.array(xi) for xi in beta])
    #print(np_beta)
    beta = [row[::-1] for row in beta]
    beta = np.array([np.array(xi) for xi in beta])
    #print(beta)
    #print(beta.shape)
    return beta

def update(alpha, beta, A, B, seq):
    n_states = alpha.shape[0]
    T = len(seq)
    P_Y_theta = np.sum(alpha*beta)
    
    gamma = []
    for n in range(n_states):
        gamma.append([])
        for t in range(T):
            gamma[n].append(alpha[n, t]*beta[n, t]/P_Y_theta)
    gamma = np.array([np.array(xi) for xi in gamma])
    #print(gamma)
    
    xi = np.zeros((n_states*n_states, T-1))
    for i in range(xi.shape[0]):
        ni = i//n_states
        nj = i%n_states
        for t in range(xi.shape[1]):
            xi[i,t] = alpha[ni, t]*A[ni, nj]*beta[nj, t]*B[nj, seq[t+1]]/P_Y_theta
    #print(xi)

    #print(gamma.shape, xi.shape)
    return gamma, xi

def normalize_row(m):
    row_sums = m.sum(axis=1)
    #print(row_sums)
    m = m / row_sums[:, np.newaxis]
    return m

def Baum_Welch(A, B, pi, seq, n_iters=100):
    T = len(seq)
    for _ in range(n_iters):
        alpha = forward(A, B, pi, seq)
        beta = backward(A, B, pi, seq)
        gamma, xi = update(alpha, beta, A, B, seq)
        
        
        newPi = gamma[:,0].reshape(-1,1)
        newA = A.copy()
        for i in range(newA.shape[0]):
            for j in range(newA.shape[1]):
                sum_gamma = np.sum(gamma[:, :T-2], axis=1)
                sum_xi = np.sum(xi[:, :T-2], axis=1)
                newA[i,j] = sum_xi[i*n_states+j]/sum_gamma[i]

        newB = B.copy()
        sum_gamma = np.sum(gamma, axis=1)
        for i in range(newB.shape[0]):
            for k in range(newB.shape[1]):
                mask = np.ones(T)
                mask = np.where(seq==k, 1, 0)
                #print(mask.shape)
                mask_gamma = gamma*mask
                sum_mask_gamma = np.sum(mask_gamma, axis=1)
                newB[i, k] = sum_mask_gamma[i]/sum_gamma[i]

        #print(newA)
        #print(newB)
        #print(newPi)
        #print(np.sum(newPi))
        #print(newPi/np.sum(newPi))
        A = normalize_row(newA)
        B = normalize_row(newB)
        pi = newPi/np.sum(newPi)
        #print("PI")
        #print(pi)
        if (_+1)%(n_iters/10)==0:
            print(_+1)
            print("Transition")
            print(A)
            print("Emission")
            print(B)
            print('Prob')
            print(pi)
            print("________________________________________________________________________________________________________________________")
    

if __name__ == '__main__':
    head_tail = info()
    n_states = 2
    n_emissions = 2
    
    pi = initPi(n_states)
    transition = initTransition(n_states)  # Ex: AA, AB...
    emission = initEmission(n_states, n_emissions)   #Ex: A0, A1, A2...
    

    pi = [[0.1],[0.9]]
    pi = np.array([np.array(xi) for xi in pi])
    trans = [[0.9, 0.1],[0.1, 0.9]]
    transition = np.array([np.array(xi) for xi in trans])
    emi = [[0.5, 0.5],[0.9, 0.1]]
    emission = np.array([np.array(xi) for xi in emi])
    #initial = {"A":0.7, "B":0.3}
    #transition = {"AA":0.9, "AB":0.1, "BA":0.1, "BB":0.9}
    #emission = {"A0":0.5, "A1":0.5, "B0":0.9, "B1":0.1}
    
    print("\n*********************** STARTING CONDITION ***********************")
    print("Initial prob:", pi.shape)
    print(pi)
    print("Transition prob:", transition.shape)
    print(transition)
    print("Emission prob:", emission.shape)
    print(emission)
    print("*********************** STARTING CONDITION ***********************\n")
    head_tail = head_tail - 1
    
    #sequence = turnObservationToSequence(head_tail)

    #print(head_tail)
    #alpha = forward(transition, emission, pi, head_tail.astype(int))
    #beta = backward(transition, emission, pi, head_tail.astype(int))
    #update(alpha, beta, transition, emission, head_tail.astype(int))
    len_sys_argv = len(sys.argv)
    print(len_sys_argv)
    if len_sys_argv == 1:
        Baum_Welch(transition, emission, pi, head_tail.astype(int))
    else:
        Baum_Welch(transition, emission, pi, head_tail.astype(int), int(sys.argv[1]))


