"""
ESE 545 Project 4

@author: Yiidng Zhang, Xiangyang Cui
"""

import matplotlib.pyplot as plt
from greedyAlgo import *
from lazyAlgo import *


def plotBenefits(G_bnft, L_bnft, k_range):
    plt.plot(k_range, G_bnft, 'b*-', label="Greedy Algorithm")
    plt.plot(k_range, L_bnft, 'r+-', label="Lazy Algorithm")
    plt.title("Benefits (Object Values) F(A)")
    plt.xlabel("k")
    plt.legend()
    plt.show()


def plotTimeStamps(G_time, L_time, k_range):
    plt.plot(k_range, G_time, 'b*-', label="Greedy Algorithm")
    plt.plot(k_range, L_time, 'r+-', label="Lazy Algorithm")
    plt.title("Running Time")
    plt.xlabel("k")
    plt.ylabel("Time /sec")
    plt.legend()
    plt.show()
    

def main(k = 50):
    print("Reading data...")
    data = readData("ml-1m/ratings.dat")
    print("Complete!\n")
    
    print("Greedy Algorithm...")
    (G_rcmd,G_bnft,G_time) = greedyAlgo(data, k)
    print("movie IDs to be recommended through Greedy Algorithm:")
    print(G_rcmd)
    
    print("\nLazy Algorithm...")
    (L_rcmd,L_bnft,L_time) = lazyAlgo(data, k)
    print("movie IDs to be recommended through Lazy Algorithm:")
    print(L_rcmd)
    
    k_range = np.array([10, 20, 30, 40, 50])
    plotBenefits(G_bnft, L_bnft, k_range)
    plotTimeStamps(G_time, L_time, k_range)


if __name__ == "__main__":
    main()