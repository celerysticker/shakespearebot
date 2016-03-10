# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:20:04 2016

@author: nancywen
"""

import json 
import numpy as np

A = []
O = []

num_states = 10

with open('hmm.txt', 'r') as f:
    for i in range(num_states):
        A.append([float(x) for x in f.readline().strip().split('\t')])
    for i in range(num_states):
        O.append([float(x) for x in f.readline().strip().split('\t')])
   
A = np.array(A)
O = np.array(O)

# read in the json dictionaries
with open('obs_dict.json') as df:
    obs_dict = json.load(df)
    
with open('rev_obs_dict.json') as df2:
    rev_obs_dict = json.load(df2)
    
