
"""
ESE 545 Project 4

@author: Yiidng Zhang, Xiangyang Cui
"""

import time
from readData import *


def greedyAlgo(data, k = 50):
    recommendation = [] # The movies selected
    benefits = [] # F(A), or objective value
    time_stamp = []
    corr_max = np.zeros((1,COL)) # initial A0, empty
    
    start = time.time()
    for t in range(1, k + 1):
        new_matrix = np.maximum(data, corr_max) 
        avg_rating = new_matrix.sum(axis=1) / COL # calculate f(A) for each row
        target_idx = np.argmax(avg_rating,axis=0) # find e*
        corr_max = new_matrix[target_idx,:] # equivlent to F(A) Union e
        recommendation.append(target_idx)
        
        if t % 10 == 0:
            benefits.append(np.sum(corr_max) / COL)
            time_stamp.append(time.time() - start)
            
    return (recommendation, benefits, time_stamp)

