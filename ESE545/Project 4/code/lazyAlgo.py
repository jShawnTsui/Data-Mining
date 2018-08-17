
"""
ESE 545 Project 4

@author: Yiidng Zhang, Xiangyang Cui
"""

import time
from readData import *

def lazyAlgo(data, k = 50):
    recommendation = [] # The movies selected
    benefits = [] # F(A), or objective value
    time_stamp = []
    
    prev_max = np.zeros((1,COL)) # initial A0, empty
    prev_benefit = 0 # F(A0) = 0
    # Iteration 1
    start = time.time()
    avg_rating = np.sum(data, axis=1) / COL # calculate F(A) for each row
    indices = np.argsort(avg_rating,axis=0)[::-1] # sort the marginal benefits
    recommendation.append(indices[0]) # pick the row with largest marginal benefit
    corr_benefit = avg_rating[indices[0]] # F(A1) = F(e*)
    corr_max = data[indices[0]] # A1 = e*
    # Subsequent iterations
    i = 1
    for t in range(2, k + 1):
        corr_delta = marginalBenefit(data, indices[i], corr_max, corr_benefit) # like Del(e2|A1)
        prev_delta = marginalBenefit(data, indices[i+1], prev_max, prev_benefit) # like Del(e3|A0)
        if corr_delta >= prev_delta:
            recommendation.append(indices[i])
            prev_max = corr_max
            corr_max = np.maximum(data[indices[i]], corr_max) # A Union e*
            prev_benefit = corr_benefit
            corr_benefit = corr_delta + corr_benefit
            i = i + 1
        else:
            data = np.maximum(data, corr_max) 
            avg_rating = np.sum(data, axis=1) / COL
            indices = np.argsort(avg_rating,axis=0)[::-1] # re_sort according to marginal benefits
            recommendation.append(indices[0])
            corr_benefit = avg_rating[indices[0]]
            corr_max = data[indices[0]]
            i = 1
        if t % 10 == 0:
            benefits.append(corr_benefit)
            time_stamp.append(time.time() - start)
    return (recommendation, benefits, time_stamp)

def marginalBenefit(data, idx, corr_max, benefit):
    return np.maximum(data[idx], corr_max).sum() / COL - benefit

    
    