
"""
ESE 545 Project 4

@author: Yiidng Zhang, Xiangyang Cui
"""

import numpy as np 

ROW = 3952 # number of distinct movies
COL = 6040 # number of distinct users

def readData(filePath):
    matrix = np.zeros((ROW, COL))
    with open(filePath, 'r') as file:
        for line in file:
            features = line.split('::')
            matrix[int(features[1])-1, int(features[0])-1] = int(features[2])
    return matrix

    