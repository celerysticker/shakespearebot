# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 23:37:56 2016

@author: nancywen
"""
from nltk.tokenize import RegexpTokenizer

# Run this line the first time
# nltk.download('punkt')

def tokenize_lines_words_only(filename):
    """ Creates a list of tokens out of the poems that only contains
    the words (no punctuation) """
    all_tokens = []
    line_count = 14
    i = 0
    tokenizer = RegexpTokenizer(r'\w+')
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip()
            try:
                int(line)
                i = 0
                continue
            except ValueError:
                pass
            if len(line) > 0:
                tokens = tokenizer.tokenize(line)
                all_tokens.append(tokens)
            i += 1
    return all_tokens

def get_words_only_str(line):
    tokenizer = RegexpTokenizer(r'\w+')
    line = line.strip()
    if len(line) > 0:
        tokens = tokenizer.tokenize(line)
    return tokens
    
# if __name__ == "__main__":   
#     tokenize_lines_words_only('data/shakespeare_reduced.txt')