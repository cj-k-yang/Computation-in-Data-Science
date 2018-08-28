import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import operator
import copy
from hmmlearn import hmm

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

def initInitial(n_states):
    state_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ini = {}
    initProb = 1.0/n_states
    for i in range(n_states):
        ini[state_key[i]] = initProb
    return ini


def initTransitionDict(n_states):
    state_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    trans = {}
    for i in range(n_states):
        initProb = sklearn.preprocessing.normalize(np.random.uniform(0, 1, n_states).reshape(-1, 1), norm='l1', axis=0).ravel()
        #print(initProb)
        for j in range(n_states):
            key = state_key[i] + state_key[j]
            trans[key] = initProb[j]
    return trans

def initEmissionDict(n_states, n_emissions):
    state_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    emission_key = "0123456789"
    emi = {}
    for i in range(n_states):
        initProb = sklearn.preprocessing.normalize(np.random.uniform(0, 1, n_emissions).reshape(-1, 1), norm='l1', axis=0).ravel()
        for j in range(n_emissions):
            key = state_key[i] + emission_key[j]
            emi[key] = initProb[j]
    return emi

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

def updateTransition(n_states, init, transition, emission, sequence):
    newTrans = transition.copy()
    tran_states = []
    for k in transition.keys():
        tran_states.append(k)
    sequence_count = count_seq(sequence)
    
    all_sequence_states_prob = {}
    for k in sequence_count.keys():
        all_sequence_states_prob[k] = {}
        for s in tran_states:
            all_sequence_states_prob[k][s]  = init[s[0]]*emission[s[0]+k[0]]*transition[s]*emission[s[1]+k[1]]

    unnormalized = []
    for s in tran_states:
        numerator, denominator = 0.0, 0.0
        for k,i in sequence_count.items():
            max_key = max(all_sequence_states_prob[k].items(), key=operator.itemgetter(1))[0]
            denominator += i*all_sequence_states_prob[k][max_key]
            numerator += i*all_sequence_states_prob[k][s]
        unnormalized.append(numerator/denominator)
    
    normalized = []
    for i in range(int(len(unnormalized)/n_states)):
        normalized.append(sklearn.preprocessing.normalize(np.array(unnormalized[i*n_states:n_states+i*n_states]).reshape(-1, 1), norm='l1', axis=0).ravel())
    
    normalized = np.array(normalized).reshape(-1,1).ravel()

    for i,s in enumerate(tran_states):
        newTrans[s] = normalized[i]
    return newTrans

def getNumeratorForUE(k, k2, d):
    wanted_emi = k[0]
    wanted_state = k[1]
    toContinue = True
    toReturn = 0.0
    keys = list(d.keys())
    #del unwanted
    for key in keys:
        for i,c in enumerate(k2):
            if c==wanted_emi and key[i]!=wanted_state:
                d.pop(key, None)
                break
        
    max_key = max(d.items(), key=operator.itemgetter(1))[0]
    return d[max_key]
                
def normalize_emi_dic(d, n_states, n_emissions):
    keys = list(d.keys())
    keys.sort()
    
    unnormalized = []
    normalized = []
    for i in range(n_states):
        unnormalized.append([])
        normalized.append([])

    #print(d)
    #print(keys)
    for i,k in enumerate(keys):
        #print(i,k)
        if (i+1)%n_emissions!=0:
            unnormalized[int(i//n_emissions)].append(d[k])
        else:
            unnormalized[int(i//n_emissions)].append(d[k])
            #print(111, i+1)
            #print(unnormalized[int(i//n_emissions)])
            normalized[int(i//n_emissions)]+=(sklearn.preprocessing.normalize(np.array(unnormalized[int(i//n_emissions)]).reshape(-1, 1), norm='l1', axis=0).ravel().tolist())
    #print(normalized)
    for i,k in enumerate(keys):
        d[k] = normalized[i//n_emissions][i%n_emissions]
            
        

def updateEmission(n_states, n_emissions, init, transition, emission, sequence):
    state_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    emission_key = "0123456789"
    emission_state_dic = {}

    all_emi = []
    for ek in emission.keys():
        all_emi.append(ek)

    all_seq = []
    for i in range(n_emissions):
        for j in range(n_emissions):
            all_seq.append(str(i)+str(j))

    tran_states = []
    for k in transition.keys():
        tran_states.append(k)

    for i in range(n_emissions):
        for j in range(n_states):
            newKey = emission_key[i]+state_key[j]
            emission_state_dic[newKey] = {}
            for k in all_seq:
                if emission_key[i] not in k:
                    continue
                emission_state_dic[newKey][k] = {}
                for s in tran_states:
                    #print(init[s[0]])
                    #print(emission[s[0]+k[0]])
                    #print(transition[s])
                    #print(emission[s[1]+k[1]])
                    emission_state_dic[newKey][k][s] = init[s[0]]*emission[s[0]+k[0]]*transition[s]*emission[s[1]+k[1]]
    newEmission = {}
    #print(emission_state_dic)
    for k,i in emission_state_dic.items():
        #wanted_emi = k[0]
        #wanted_state = k[1]
        newEmissionKey = k[::-1]

        for s in tran_states:
            numerator, denominator = 0.0, 0.0
            for k2,i in emission_state_dic[k].items():
                max_key = max(emission_state_dic[k][k2].items(), key=operator.itemgetter(1))[0]
                denominator += emission_state_dic[k][k2][max_key]
                numerator += getNumeratorForUE(k, k2, emission_state_dic[k][k2].copy())
                print(denominator, numerator)
        newEmission[newEmissionKey] = numerator/denominator
    normalize_emi_dic(newEmission, n_states, n_emissions)
    return newEmission
    #print("____________________________")
    #print(emission_state_dic)
    #print(newEmission)

def Baum_Welch(n_states, n_emissions, init, transition, emission, sequence, n_iters=1000):
    for i in range(n_iters):
        newTran = updateTransition(n_states, init, transition, emission, sequence)
        newEmi = updateEmission(n_states, n_emissions, init, transition, emission, sequence)
        transition = newTran
        emission = newEmi
        print(emission)
        if (i+1)%(n_iters/10)==0:
            print(i+1)
            print("Transition")
            print(transition)
            print("Emission")
            print(emission)

if __name__ == '__main__':
    head_tail = info()
    n_states = 2
    n_emissions = 2
    
    #initial = initInitial(n_states)
    #transition = initTransitionDict(n_states)  # Ex: AA, AB...
    #emission = initEmissionDict(n_states, n_emissions)   #Ex: A0, A1, A2...
    
    initial = {"A":0.7, "B":0.3}
    transition = {"AA":0.9, "AB":0.1, "BA":0.1, "BB":0.9}
    emission = {"A0":0.5, "A1":0.5, "B0":0.9, "B1":0.1}


    print("\n*********************** STARTING CONDITION ***********************")
    print("Initial prob:", initial)
    print("Transition prob:", transition)
    print("Emission prob:", emission)
    print("*********************** STARTING CONDITION ***********************\n")
    head_tail = head_tail - 1
    #print(head_tail)

    sequence = turnObservationToSequence(head_tail)
    #print(sequence)
    model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100).fit(head_tail.reshape(-1,1))
    print(model.transmat_)
    print(model.startprob_)
    print(model.covars_)
    #Baum_Welch(n_states, n_emissions, initial, transition, emission, sequence, 100)
