import numpy as np 
import matplotlib.pyplot as plt
import time
import resource
"""
Problem 5
"""
M = 1000
ROW = 5
BAND = int(M / ROW)
PRIME = 2900017

def readCSVFile():
    """Read rating.csv and form the raw matrix"""
    raw_matrix = np.load('ratings_np.npy')
    return raw_matrix

def jacSim4CmprsMat(user_col, user0_index, user1_index):
    intersection = np.intersect1d(user_col[user0_index], user_col[user1_index])
    union = np.union1d(user_col[user0_index], user_col[user1_index])
    return intersection.size/np.float(union.size)

def compressMatrix(raw_matrix):
    users = np.nonzero(np.max(raw_matrix, 0))[0]
    user_col = []
    for i in users:
        user_col.append(np.nonzero(raw_matrix[:, i])[0])
    return (users, user_col)

def generateSignature(user_col):
    length = user_col.__len__()
    k = np.random.randint(1, PRIME, (M, 1))
    b = np.random.randint(1, PRIME, (M, 1))
    signature = np.zeros((length, M), dtype='int64')
    i = 0
    for user in user_col:
        signature[i, :] = np.mod(k * user + b, PRIME).min(1)
        i += 1
    return signature

def signatureHash(signature):
    length = signature.shape[0]
    k = np.random.randint(1, PRIME, M)
    b = np.random.randint(1, PRIME, M)
    sig_hash = np.mod(k * signature + b, PRIME)
    final_sig  = np.zeros((BAND, length), dtype='int64')
    window = np.arange(ROW)
    for i in range(BAND):
        final_sig[i, :] = sig_hash[:, window + i * ROW].sum(1)
    return final_sig

def findNeighbors(user_col, users, final_sig, query):
    q_index = np.where(users == query)[0]
    if q_index.size == 0:
        return (-1, -1)
    q_index = q_index[0]
    column = final_sig.shape[1]
    final_sig = np.mat(final_sig)
    compare = sum(final_sig == final_sig[:, q_index], 0)
    indices = np.array(np.argsort(-compare))[0]
    compare = np.array(-np.sort(-compare))[0]
    if compare[1] == 0:
        return (-1,-1)
    for i in indices[1:]:
        if jacSim4CmprsMat(user_col, i, q_index) > 0.4:
            return (users[i], jacSim4CmprsMat(user_col, i, q_index))
    return (-1,-1)

def main():
    """This is where project starts"""
    query = int(input('Enter a number between 0~138432:')) + 1
    
    print('Reading file...')
    raw_matrix = readCSVFile()
    #users = np.load('users.npy')
    print('Sparse Matrix Compressing...')
    (users, user_col) = compressMatrix(raw_matrix)
    if query not in users:
        print('Didn\'t find neighbor with similarity higher than 0.4.')
        return
    print('Sparse Matrix Compress Completed...')
    #print('TIME: ', time.time()-start)
    
    
    #user_col = np.load('user_col.npy')
    
    #signature = np.load('signature.npy')
    print('Signature Generating.')
    signature = generateSignature(user_col)
    print('Signature Generation Completed.')
    #print('TIME: ', time.time()-start)
    final_sig = signatureHash(signature)
    print('Signature Hash Completed.')
    #print('TIME: ', time.time()-start)

    #final_sig = np.load('final_sig.npy')
    print('Searching for the nearest neighbor.')
    start = time.time()
    (neighbor, prop) = findNeighbors(user_col, users, final_sig, query)
    if neighbor == -1:
        print('Didn\'t find neighbor with similarity higher than 0.4.')
    else:
        print('The nearest Neighbor is:', neighbor)
        print('The similarity is:',prop)
    print('TIME: ', time.time()-start)
    #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == '__main__':
    main()
