# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 00:17:31 2016

@author: nancywen

"""

from utils import nsyl
from utils import get_syllables_str
import random 

import json 

# read in the json dictionaries
with open('obs_dict.json') as df:
    obs_dict = json.load(df)  # maps words to nums
    
with open('rev_obs_dict.json') as df2:
    rev_obs_dict = json.load(df2) # maps nums to words
    
with open('rhyme_dict.json') as df3:
    rhyme_dict = json.load(df3)    
    
def generate_line_backwards(line_len, last_word):
    '''
    Given the last word of a line and the number of syllables in a line
    Return a line of the sonnet
    '''
    line = [last_word]
    num_syllables = nsyl(last_word)
    while num_syllables is not line_len:
        # randomly choose a word
        candidate_len =  -1 
        while candidate_len is -1:
            candidate_word = random.choice(obs_dict.keys())
            candidate_len = nsyl(candidate_word)
        if num_syllables + candidate_len <= line_len:
            line.append(candidate_word)
            num_syllables += candidate_len
    line.reverse()
    return line
    
def generate_line(line_len, first_word):
    '''
    TODO: Given the first word of a line and the number of syllables in a line
    Return a line of the sonnet
    '''
    pass
        
def pick_last_word(line_index, sonnet):
    '''
    Given the line index, and the poem generate so far, 
    Return a last word that rhyme with the appropriate line
    '''
    if line_index is 0 or line_index is 1 or line_index is 4 or line_index is 5 or\
       line_index is 8 or line_index is 9 or line_index is 12:
        # randomly select last word from rhyming dictionary
           last_word = random.choice(rhyme_dict.keys())
           
    else:
        if line_index is 13:
            rhyme_line = sonnet[line_index - 1]
        else:
            rhyme_line = sonnet[line_index - 2]
        rhyme_word = rhyme_line[-1]
        last_words = rhyme_dict[rhyme_word] # get a list of all rhyming words
        last_word = random.choice(last_words) # perhaps, ensure not same word   
         
    #print last_word
    return last_word
    
def pick_first_word(line_index, sonnet):
    '''
    TODO: Given the line index and the poem generated so far
    Return the first word of the next line
    '''
    pass
    
def print_sonnet(sonnet):
    '''
    Given the sonnet in list form, 
    Print sonnet to screen
    '''
    
    for l, line in enumerate(sonnet):
        print_line = ''
        for w, word in enumerate(line):
            # first word is capitalized
            if w is 0:
                print_line += word.capitalize() + " "
            else:
                print_line += word + " "
        if l is 12 or l is 13:
            print "  " + print_line
        else:
            print print_line

def sonnet_to_file(sonnet):
    """
    TODO: Prints the generate sonnet to a text file
    """
    pass

def make_sonnet(rhyming):
    '''
    Given whether it's rhyming, generates a sonnet
    '''
    sonnet = []   
    num_words = len(obs_dict) # 3433
    num_rhyme_words = len(rhyme_dict) # 1011
    num_lines = 14
    line_len = 10
    
    if rhyming:
        for i in range(num_lines):
            # randomly choose a last word
            last_word = pick_last_word(i, sonnet)
                    
            curr_line = generate_line_backwards(line_len, last_word)
            sonnet.append(curr_line)
    else:
        for i in range(num_lines):
            # randomly choose a last word
            first_word = pick_first_word(i, sonnet)
            curr_line = generate_line(line_len, first_word)
            sonnet.append(curr_line)  
    return sonnet

sonnet = make_sonnet(True)     
print_sonnet(sonnet)
