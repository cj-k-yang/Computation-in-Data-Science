import random
from random import randrange, shuffle, uniform
from sys import argv
import numpy as np
import operator

random.seed(8787)


def count_consecutive_char(text, d):
    text_len = len(text)
    for i in range(text_len-1):
        consecutive_char = text[i] + text[i+1]
        #print(consecutive_char)
        d[consecutive_char] += 1

def initialize_char_dic(wantedChar, d):
    for i in wantedChar:
        for j in wantedChar:
            d[i+j] = 1

def initialize_prob_dic(count_d, prob_d):
    keys = []
    for k in count_d.keys():
        keys.append(k)
    
    for i in range(27):
        total = 0
        for j in range(i*27, 27+i*27):
            total += count_d[keys[j]]
        for j in range(i*27, 27+i*27):
            prob_d[keys[j]] = 1.0+count_d[keys[j]]/float(full_text_len)
            #print(count_d[keys[j]])


def decypher(text, key, wantedChar="abcdefghijklmnopqrstuvwxyz "):
    return text.translate(str.maketrans(key, wantedChar))

def randomSwap(keys):
    index1 = np.random.randint(0, 27)
    index2 = np.random.randint(0, 27)
    while index2 == index1:
        index2 = np.random.randint(0, 26)
    tempChar = keys[index1]

    tempKeys = list(keys)
    tempKeys[index1] = tempKeys[index2]
    tempKeys[index2] = tempChar
    
    return "".join(tempKeys)


def load(filename):
    f = open(filename, 'r', encoding='utf-8')
    text = f.read()
    f.close()
    return text

def calculateP(text, keys, prob_d): # paper 1
    newText = decypher(text, keys)
    #print(newText)
    textLen = len(newText)
    #print(textLen)
    p = 1.0
    for i in range(textLen-1):
        consecutive_char = newText[i] + newText[i+1]
        p *= prob_d[consecutive_char]
        #print(p)
        #print(prob_d[consecutive_char])
    return p

def calculateP2(text, keys, r_count_d): # paper 2
    newText = decypher(text, keys)

    f_count_dict = {}
    initialize_char_dic(wanted27char, f_count_dict)
    count_consecutive_char(newText, f_count_dict)
    #print(newText)
    textLen = len(newText)
    #print(textLen)
    p = 0.0
    for i in range(textLen-1):
        consecutive_char = newText[i] + newText[i+1]
        p += np.log10(r_count_d[consecutive_char])*f_count_dict[consecutive_char]
        #print(p)
        #print(prob_d[consecutive_char])
    return p

def calculateP3(text, keys, r_count_d): # BEST
    p=0
    newText = decypher(text, keys)
    for i in range(len(text)-1):
        consecutive_char = newText[i] + newText[i+1]
        p = p + np.log(r_count_d[consecutive_char]/full_text_len)
    return p

def MCMC(text, keys, prob_d, count_d, iterations=1000000):
    #P = calculateP(text, keys, prob_d)
    #P = calculateP2(text, keys, count_d)
    P = calculateP3(text, keys, count_d)
    bestP = -1000000.0
    bestKeys = {}
    for i in range(iterations):
        #print(i)
        newKeys = randomSwap(keys)
        #newP = calculateP(text, newKeys, prob_d)
        #newP = calculateP2(text, newKeys, count_d)
        newP = calculateP3(text, newKeys, count_d)
        if newP >= P:
            keys = newKeys
            P = newP
        else:
            #r = newP/P
            #r = np.exp(P-newP)
            r = np.exp(newP-P)
            #print(r)
            uni = uniform(0,1)
            #print(r)
            if uni < r:
                keys = newKeys
                P = newP
        #print(keys)
        if newP >= bestP:
            bestP = newP
            bestKeys = newKeys
        if (i+1)%10000==0:
            print(i+1, P, bestP)
    return keys, bestKeys


def print_text(text):
    print("*****************************************")
    print(text)
    print("*****************************************")

def count_alphabet(text):
    d = {}

    for c in wanted27char:
        d[c] = 0
    total = len(text)
    for i in text:
        d[i] += 1

    for i in d.keys():
        print(i, d[i]/float(total))
    
    return d
    #print(sorted_x)

def sort_alphabet(d1, d2):
    
    sorted_x1 = sorted(d1.items(), key=operator.itemgetter(1))
    sorted_x2 = sorted(d2.items(), key=operator.itemgetter(1))
    for i in range(27):
        print(sorted_x1[26-i][0], sorted_x2[26-i][0])

if __name__ == '__main__':
    war_and_peace = load("war and peace.txt")
    #war_and_peace = load("domain.txt")
    wanted27char = "abcdefghijklmnopqrstuvwxyz "

    war_and_peace = ''.join([s for s in war_and_peace if s in wanted27char])
    war_and_peace = " ".join(war_and_peace.split())
    full_text_len = len(war_and_peace)
    count_dict = {}
    initialize_char_dic(wanted27char, count_dict)
    count_consecutive_char(war_and_peace, count_dict)

    prob_dict = {}
    initialize_prob_dic(count_dict, prob_dict)

    tempKey = list(wanted27char)
    shuffle(tempKey)
    key = "".join(tempKey)
    
    wanted_text = load("decode.txt")
    print("to be decoded:\n")
    print_text(wanted_text)
    #my_key = "fbor hpackswjdemvgzqylixntu"
    
    
    k, bestk = MCMC(wanted_text, key, prob_dict, count_dict, 100000)
    #k, bk = MCMC(wanted_text, key, prob_dict, count_dict, 1)
    
    print("last key:", k)
    print_text(decypher(wanted_text, k))
    print("best key:", bestk)
    print_text(decypher(wanted_text, bestk))
