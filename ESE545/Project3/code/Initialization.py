from readData import *

def randInitialize(data, K):
    rand_centroids = np.random.randint(data.shape[0], size=K)
    return rand_centroids

def chooseNextCenter(data, kmpp_centroids):
    dist2 = np.array([min([np.linalg.norm(x-data[c])**2 for c in kmpp_centroids]) for x in data])
    probs = dist2 / dist2.sum()
    cumprobs = probs.cumsum()
    r = np.random.random()
    return np.where(cumprobs >= r)[0][0]


def kmppInitialize(data, K):
    kmpp_centroids = [np.random.randint(data.shape[0])]
    for i in np.arange(1, K):
        kmpp_centroids.append(chooseNextCenter(data, kmpp_centroids))
    return np.array(kmpp_centroids)

def nuKmc2Initialize(data, K, m=200):
    kmc2_centroids = [np.random.randint(data.shape[0])]
    for i in np.arange(1, K):
        x = np.random.randint(data.shape[0])
        x_dist2 = min([np.linalg.norm(data[x]-data[c])**2 for c in kmc2_centroids])

        for j in np.arange(1, m):
            y = np.random.randint(data.shape[0])
            y_dist2 = min([np.linalg.norm(data[y]-data[c])**2 for c in kmc2_centroids])
            if x_dist2 == 0 or y_dist2/x_dist2 > np.random.uniform():
                x = y
                x_dist2 = y_dist2

        kmc2_centroids.append(x)

    return np.array(kmc2_centroids)

def initialize(name, data, K):
    if name == 'random':
        return randInitialize(data, K)
    if name == 'kmpp':
        return kmppInitialize(data, K)
    if name == 'kmc2':
        return nuKmc2Initialize(data, K)

def main():
    K = 10
    data = np.load('data.npy')
    rand_centroids = randInitialize(data, K)
    print(rand_centroids)
    kmc2_centroids = nuKmc2Initialize(data, K)
    print(kmc2_centroids)

if __name__ == '__main__':
    main()
