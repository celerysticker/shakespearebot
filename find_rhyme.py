# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 23:52:54 2016

@author: nancywen
"""

'''
Rhyme scheme of Shakespearean sonnet:
A - 0
B - 1
A - 2
B - 3
C - 4
D - 5
C - 6
D - 7
E - 8
F - 9
E - 10
F - 11
G - 12
G - 13
'''


from preprocess_words_only import tokenize_lines_words_only

def make_rhyme_dictionary(filename):
    rhyme_dict = dict()
    last_words = []
    tokens = tokenize_lines_words_only(filename)
    for line in tokens:
        last_words.append(line[len(line) - 1])
        
    for i, word in enumerate(last_words):
        if i%14 == 12:
            if word not in rhyme_dict:
                rhyme_dict[word] = [last_words[i + 1]]
            else:
                rhyme_dict[word].append(last_words[i + 1])
        elif i%14 == 13:
            if word not in rhyme_dict:
                rhyme_dict[word] = [last_words[i - 1]]
            else:
                rhyme_dict[word].append(last_words[i - 1])
        elif i%14 == 0 or i%14 == 1 or i%14 == 4 or i%14 == 5 or i%14 == 8 or i%14 == 9:  
            if word not in rhyme_dict:
                rhyme_dict[word] = [last_words[i + 2]]
            else:
                rhyme_dict[word].append(last_words[i + 2])
        else:
            if word not in rhyme_dict:
                rhyme_dict[word] = [last_words[i - 2]]
            else:
                rhyme_dict[word].append(last_words[i - 2])
    return last_words
  
rhyme_dict = make_rhyme_dictionary('data/shakespeare_reduced.txt')