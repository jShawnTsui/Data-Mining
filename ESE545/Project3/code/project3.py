from MiniBatchKMeans import *

def plot(Kmax, Kmin, Kmean, maxes, mines, meanes, K_range):
    plt.figure(1)
    plt.plot(K_range, Kmax, 'r-', label='K-Mean++')
    plt.plot(K_range, maxes, 'b-', label='Random')
    plt.title('Max Distance')
    plt.xlabel('K')
    plt.ylabel('Distance')
    plt.legend()
    plt.figure(2)
    plt.plot(K_range, Kmin, 'r-', label='K-Mean++')
    plt.plot(K_range, mines, 'b-', label='Random')
    plt.title('Min Distance')
    plt.xlabel('K')
    plt.ylabel('Distance')
    plt.legend()
    plt.figure(3)
    plt.plot(K_range, Kmean, 'r-', label='K-Mean++')
    plt.plot(K_range, meanes, 'b-', label='Random')
    plt.title('Mean Distance')
    plt.xlabel('K')
    plt.ylabel('Distance')
    plt.legend()
    plt.show()

def plotMean(Kmean, Rmean, T):
    plt.plot(T, Kmean, 'r-', label='K-Mean++')
    plt.plot(T, Rmean, 'b-', label='Random')
    plt.title('Mean Distance K = 100 B = 10')
    plt.xlabel('T')
    plt.ylabel('Distance')
    plt.legend()
    plt.show()

def main():
    print('Reading data...')
    
    file_name = './R6/ydata-fp-td-clicks-v1_0.20090501.gz'
    data = readData(file_name)

    print('Convergence of mean distance with same K and T, but different initializations')
    K = 100
    B = 10
    T = 200
    print('Random method:')
    Rmean = miniBatchKMeans('random', data, K, B, T, mode=True)
    print('K-Mean++ method:')
    Kmean = miniBatchKMeans('kmc2', data, K, B, T, mode=True)
    plotMean(Kmean, Rmean, np.arange(0, T, 10))

    print('For different K, the changes of max, min and mean distances of two methods')
    # For different K
    B = 1000
    T = 100
    K_range = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 150, 200, 250, 300, 400, 500])
    Kmax = list()
    Kmin = list()
    Kmean = list()
    Rmax = list()
    Rmin = list()
    Rmean = list()

    print('Random method:')
    name = 'random'
    for K in K_range:
        ctr_data = miniBatchKMeans(name, data, K, B, T)
        (maximun, minimun, mean) = calculate(data, ctr_data)
        Rmax.append(maximun)
        Rmin.append(minimun)
        Rmean.append(mean)
        print('\t', K/5,'% finished')

    print('K-Mean++ method:')
    name = 'kmc2'
    for K in K_range:
        ctr_data = miniBatchKMeans(name, data, K, B, T)
        (maximun, minimun, mean) = calculate(data, ctr_data)
        Kmax.append(maximun)
        Kmin.append(minimun)
        Kmean.append(mean)
        print('\t', K/5,'% finished')

    plot(Kmax, Kmin, Kmean, Rmax, Rmin, Rmean, K_range)

if __name__ == '__main__':
    main()