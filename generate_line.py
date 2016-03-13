# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 12:53:19 2016

@author: nancywen
"""
import os
import random
import numpy as np
from utils import nsyl


def main():
    for i in range(14):
        make_line()

def make_line():    
    with open('ten_init.txt') as f:
        matrix = f.readlines()[0]
        initial_state_probs = eval(matrix)
    
    with open('ten_A.txt') as f:
        matrix = f.readlines()[0]
        A = eval(matrix)
    
    with open('ten_O.txt') as f:
        matrix = f.readlines()[0]
        O = eval(matrix)
    
    max_iter = 10 # number of hidden states to  generate 
    #first_word = "love"
    #first_word = 
    #first_state = find_state_of_word(first_word, O)
    
    state_path = find_state_path(max_iter, A, initial_state_probs, initial_state_probs)
    emissions = get_emissions(state_path, O)
    #print state_path
    prettyprint(emissions)
    
def find_state_of_word(word, OM):
    '''Given a word, find highest probability state of a word'''
    max_state = 0
    max_prob = 0
    for i in range(len(OM)):
        if OM[i][word] > max_prob:
            max_state = i
            max_prob = OM[i][word]
    return max_state

def find_state_path(max_iter, A, initial_state_probs, init_state):   
    n = len(A)
    #current_state = random.randint(0,9)
    current_state = random_pick(init_state.keys(), init_state.values())
    state_path = [current_state]
    
    for i in range(max_iter - 1):
        current_probs = A[current_state]
        #best_state = np.argmax(current_probs)
        next_state = random_pick(range(n), current_probs)
        state_path.append(next_state)
        current_state = next_state

    return state_path

def get_emissions(state_seq, OM):
    ''' Given state sequence, return sequence of emissions '''
    emission = []
    num_syllables = 0
    for e, state in enumerate(state_seq):
        # OM is a list of dictionaries
        max_obs = random_pick(OM[state].keys(), OM[state].values())
        emission.append(max_obs)
    return emission

def random_pick(some_list, probabilities):
    ''' Probabilistic random picking'''
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability: 
            break
    return item

def prettyprint(arr):
    print ' '.join(arr)

if __name__ == "__main__":
    main()
