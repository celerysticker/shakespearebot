# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 00:17:31 2016

@author: nancywen

"""

from utils import nsyl
from utils import get_syllables_str
from ngram_model import generate
from ngram_model import build_model
from ngram_model import load_data
import random 
import json 

# read in the json dictionaries
with open('obs_dict.json') as df:
    obs_dict = json.load(df)  # maps words to nums
    
with open('rev_obs_dict.json') as df2:
    rev_obs_dict = json.load(df2) # maps nums to words
    
# count the number of words not in the cmu dictionary
'''
counter = 0
for word in obs_dict:
    if nsyl(word) == -1:
        counter += 1
'''   

def generate_line(line_len):
    line = ''
    num_syllables = 0 
    while num_syllables is not line_len:
        # randomly choose a word
        candidate_len =  -1 
        while candidate_len is -1:
            i = random.randint(0, num_words - 1)
            candidate_word = rev_obs_dict[str(i)]
            candidate_len = nsyl(candidate_word)
        if num_syllables + candidate_len <= line_len:
            line += candidate_word + ' '
            num_syllables += candidate_len
    return line
        
def print_line(line):
    print_line = ''
    for w, word in enumerate(line):
        print_line += word + " "
    print print_line +  "\n"

            
fname = 'data/shakespeare.txt'
n = 2  # increase order of the ngram to generate closely to the text (no higher than 4)
data, obs_dict = load_data(fname)
model = build_model(data, n)

  
num_words = len(obs_dict)
first_len = 5
second_len = 7
third_len = 5
'''
# Generate randomly
first_line = generate_line(first_len)
second_line = generate_line(second_len)
third_line = generate_line(third_len)
'''
# Generate using n-gram model
first_line = generate(model, n, first_len)
second_line = generate(model, n, second_len, tuple(['my','love']))
third_line = generate(model, n, third_len)

print_line (first_line)
print_line (second_line)
print_line (third_line)
