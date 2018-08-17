import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from Initialization import *

def selectBatch(row, B):
    return np.random.randint(row, size=B)

def cacheClusters(b_data, ctr_data):
    return np.array([np.argmin(np.linalg.norm(x-ctr_data, axis=1)) for x in b_data])

def calculate(data, ctr_data):
    test_data = data[np.random.randint(data.shape[0], size=10000)]
    dist = np.array([min(np.linalg.norm(x-ctr_data, axis=1)) for x in test_data])
    return (np.amax(dist), np.amin(dist), np.mean(dist))

def miniBatchKMeans(name, data, K, B, T, mode=False):
    (row, column) = data.shape
    MEAN = list()
    # Initialize
    centroids = initialize(name, data, K)
    ctr_data = data[centroids]
    # Counts for each center
    v = np.zeros(K, dtype=int)

    for t in np.arange(T):
        batch = selectBatch(row, B)
        b_data = data[batch]
        batch2center = cacheClusters(b_data, ctr_data)
        for i in np.arange(B):
            c = batch2center[i]
            v[c] += 1
            eta = 1/v[c]
            ctr_data[c] = (1 - eta) * ctr_data[c] + eta * b_data[i]
        if mode and t%10 == 0:
            print('\t', t/2, '% finished')
            (_, _, mean) = calculate(data, ctr_data)
            MEAN.append(mean)
    if mode:
        return MEAN
    return ctr_data

def main():
    pass

if __name__ == '__main__':
    main()
