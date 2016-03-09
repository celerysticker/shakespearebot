# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 23:37:56 2016

@author: nancywen
"""

import nltk

# Run this line the first time
# nltk.download('punkt')

def tokenize_lines_words_only(filename):
	all_tokens = []
	line_count = 14
	i = 0
	with open(filename) as f:
		for line in f.readlines():
			line = line.strip()
			try:
				int(line)
				i = 0 # reset line count
				continue
			except ValueError:
				pass

			if len(line) > 0:
				tokens = nltk.word_tokenize(line)
				all_tokens.append(tokens)
			i += 1
	return all_tokens

print tokenize_lines_words_only('data/shakespeare.txt')