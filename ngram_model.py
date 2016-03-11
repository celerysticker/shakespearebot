# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 08:49:19 2016

@author: nancywen

Reference: http://www.decontextualize.com/teaching/rwet/n-grams-and-markov-chains/
"""

import numpy as np
import json 
from preprocess_words_only import tokenize_lines_words_only
from utils import nsyl
from utils import get_syllables_str
import random


def main():
    fname = 'data/shakespeare.txt'
    n = 3
    data, obs_dict = load_data(fname)
    model = build_model(data, n)
    #print_ngrams(model)
    
    seed = tuple(['my','love'])
    seed = None
    line_len = 10
    
    line = generate(model, n, line_len)
    print line

def generate(model, n,  line_len, seed=None, max_iterations=100):
    ''' Given ngram model and the number of syllables, 
        generates a line
        Optional: seed with first ngram
    '''
    # Randomly choose seed from all possible seeds
    if seed is None:
        seed = random.choice(model.keys())
    line = list(seed)
    current = tuple(seed)
    counter = 0
    num_syllables = 0
    for word in line:
        if nsyl(word) is -1: # hacky workaround for words not in dictionary
            num_syllables += 1
        else:
            num_syllables += nsyl(word)
    while num_syllables < line_len: # reset
        if counter > max_iterations:
            print "reset counter"
            seed = random.choice(model.keys())
            line = list(seed)
            current = tuple(seed)
            counter = 0
            num_syllables = 0
            for word in line:
                if nsyl(word) is -1: # hacky workaround for words not in dictionary
                    num_syllables += 1
                else:
                    num_syllables += nsyl(word)
        if current in model:
            possible_next_words= model[current]
            next_word = random.choice(possible_next_words)
            if nsyl(next_word) is -1:
                next_syllable = 1
            else:
                next_syllable = nsyl(next_word)
            
            if num_syllables + next_syllable <= line_len: 
                num_syllables += next_syllable
                line.append(next_word)
                print line
                print num_syllables
                current = tuple(line[-n:]) # returns last n words
                
        else:
            seed = random.choice(model.keys())
            line = list(seed)
            current = tuple(seed)
            counter = 0 
            num_syllables = 0
            for word in line:
                if nsyl(word) is -1: # hacky workaround for words not in dictionary
                    num_syllables += 1
                else:
                    num_syllables += nsyl(word)
                
    return line
        


def build_model(data, n):
    ''' Returns a dictionary of n-grams and the word that follow the n-grams'''
    model = dict()
    for line in data:
        for i in range(len(line) - n):
            ngram = tuple(line[i:i+n])
            next_word = line[i+n]
            if ngram in model:
                model[ngram].append(next_word)
            else:
                model[ngram] = [next_word]
    return model

def count_ngrams(data, n):
    ''' Returns a dictionary of n-grams and their frequency '''
    ngrams = dict()
    for line in data:
        for i in range(len(line) - n + 1):
            ngram = tuple(line[i:i+n]) # lists are not hashable, convert to tuple
            if ngram in ngrams:
                ngrams[ngram] += 1
            else:
                ngrams[ngram] = 1
    return ngrams
    
def print_ngrams(ngrams):
    ''' Prints out ngrams with frequency at least two '''
    for ngram in ngrams:
        if ngrams[ngram] > 1:
            word_ngram = ' '.join(ngram) 
            ngram_frequency = ngrams[ngram]
            print word_ngram + ": " + str(ngram_frequency)
        
def find_pairs(data):
    ''' Returns a dictionary of all pairs of words and their frequency '''
    pairs = dict()
    for line in data:
        for i in range(len(line) - 1):
            pair = tuple(line[i:i+2]) # lists are not hashable, convert to tuple
            if pair in pairs:
                pairs[pair] += 1
            else:
                pairs[pair] = 1
    return pairs
    
def load_data(fname):
    # tokenize data with preprocess
    data = tokenize_lines_words_only(fname)

    num_examples = len(data) # 2155 Shakespeare sonnets
    print "Num of training examples: " + str(num_examples)
    
    obs_dict = dict() #initialize the dictionary of observed words
    num_obs = 0
    
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
    
    print "Num of distinct obs tokens: " + str(num_obs)
    
    return data, obs_dict 
    
    
    
if __name__ == "__main__":
    main()