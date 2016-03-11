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
        
    
num_words = len(obs_dict)
first_len = 5
second_len = 7
third_len = 5
first_line = generate_line(first_len)
second_line = generate_line(second_len)
third_line = generate_line(third_len)

print '(5) ' + first_line
print '(7) ' + second_line
print '(5) ' + third_line
