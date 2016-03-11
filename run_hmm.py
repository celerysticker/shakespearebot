# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:16:37 2016

@author: nancywen
"""
import numpy as np
import json 
from preprocess_words_only import tokenize_lines_words_only
from hmm import train_hmm


def main():
    fname = 'data/shakespeare.txt'
    train_data, obs_dict = load_data(fname)
    
    make_dict(obs_dict)
    
    # pick the number of hidden states
    num_states = 10
    num_obs = len(obs_dict)
    
    # initialize the A, O matrices
    A, O = init_matrix(num_states, num_obs) 
    print "Dimensions of A: {0}".format(A.shape)
    print "Dimension of O: {0}".format(O.shape)
    
    # train hmm
    print "..........Begin training HMM.........."
    #A, O = train_hmm(train_data, num_states, num_obs, A, O)
    print "..........Complete training HMM.........."
    
    A_str = print_matrix(A) # write hmm to file
    O_str = print_matrix(O)
    with open('hmm.txt', 'w') as f:
        f.write(A_str)
        f.write(O_str)
        
def make_dict(obs_dict):
    #save dictionary to file
    with open ('obs_dict.json', 'w') as fp:
        json.dump(obs_dict, fp)
        
    rev_obs_dict = dict()
    for obs in obs_dict:
        if obs_dict[obs] not in rev_obs_dict:
            rev_obs_dict[obs_dict[obs]] = obs 
        
    #save dictionary to file
    with open ('rev_obs_dict.json', 'w') as fp2:
        json.dump(rev_obs_dict, fp2) 
        
def load_data(fname):
    # tokenize data with preprocess
    data = tokenize_lines_words_only(fname)

    num_examples = len(data) # 2155 Shakespeare sonnets
    print "Num of training examples: " + str(num_examples)
    
    obs_dict = dict() #initialize the dictionary of observed words
    num_obs = 0
    train_data = []
    
    # Make a dictionary of all observed tokens (e.g. words, punctuation)
    for i in range(num_examples):
        line = data[i]
        obs_line = []
        for j in range(len(line)):
            word = line[j]
            if word not in obs_dict:
                obs_dict[word] = num_obs
                num_obs += 1
            obs_line.append(obs_dict[word])
        train_data.append(obs_line)
    
    print "Num of distinct obs tokens: " + str(num_obs)
    
    return train_data, obs_dict 
    
def init_matrix(num_states, num_obs):
    """Returns randomly initialized A and O matrices"""
    A = np.random.uniform(size=(num_states, num_states))
    for row in A:
        row /= np.sum(row)
        
    O = np.random.uniform(size=(num_states, num_obs))
    for row in O:
        row /= np.sum(row)
       
    # make sure that there are no zero values in the initialization
    if np.count_nonzero(A) != num_states**2 or np.count_nonzero(O) != num_states * num_obs:
        print "Initialize again"
        init_matrix(num_states, num_obs)
    else:
        return A, O   
        
def print_matrix(matrix):
    matrix_str = ''
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix_str += str("{0:.5f}".format(matrix[i][j])) + '\t'
        matrix_str = matrix_str[:-2] + '\n'
    return matrix_str
    
if __name__ == "__main__":
    main()
