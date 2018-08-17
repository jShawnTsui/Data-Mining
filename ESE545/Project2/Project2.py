"""
ESE 545 - Project 2
@author: Yiding Zhang, Xiangyang Cui
"""

import csv
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import time
import math
import matplotlib.pyplot as plt

L = 0.01 # Lambda 
B = 100 # Mini_batch
T = 500 # Time of iterations
epson = 1E-6 # A small number to counter numerical instabiltiy
    
def readStopWords():
    '''
    Read stop words from given file
    Please put the file in the same folder of this script
    
    Return:
        List of stop words
    '''
    with open('stopwords.txt', 'r') as text:
        stopwords = text.read().split()
    return stopwords


def readCleanData(file_name):
    '''
    Read raw data from files, extract features
    
    Parameters:
        file_name: The path of the file to read
    Return:
        features: List of feature words
        np.array(sentiments): Array of sentiments
    '''
    start = time.time()
    stopwords = readStopWords()
    sentiments = []
    features = []
    with open(file_name, 'r') as f:
        reader = csv.reader(f)

        for line in reader:
            feature = []
            sentiments.append(-1 if line[0] == '0' else 1);

            tweet = line[5].lower()
            tweet = re.sub(r'((https?:\/\/)|(www\.))\w+(\.\w+)+(\/\w+)*', ' ', tweet)
            tweet = re.sub(r'@\S+', ' ', tweet)
            tweet = re.sub(r'[^a-zA-Z]', ' ', tweet)

            last_word = ''
            
            for word in tweet.split():
                if word != last_word:
                    if word not in stopwords:
                        feature.append(word)
                        last_word = word
            features.append(feature)

    print "Time = " + str(time.time()-start)
    return (features, np.array(sentiments))


def geneFeatureMatrix(features):
    '''
    Generate feature matrix from the list of features
    
    Parameters:
        features: List of feature words
    Return:
        feature_matrix: The feature matrix for training
        word_bag: List of indexed feature words
    '''
    start = time.time()

    cv = CountVectorizer()
    features_join = [' '.join(row) for row in features]
    feature_matrix = cv.fit_transform(features_join)
    word_bag = cv.get_feature_names()

    print "Feature matrix shape =", feature_matrix.shape
    print "Time = ", time.time()-start
    return (feature_matrix, word_bag)


def geneTestingMatrix(tst_feat, word_bag):
    '''
    Generate a feature matrix for test case
    
    Parameters:
        tst_feat: List of testing feature words
        word_bag: List of Indexed feature words
        
    Return:
        tst_matrix: the feature matrix for testing
    '''
    start = time.time()
    cv = CountVectorizer(vocabulary = word_bag)
    features_join = [' '.join(row) for row in tst_feat]
    
    tst_matrix = cv.fit_transform(features_join)
    print "Testing matrix shape =", tst_matrix.shape
    print "Time = ", time.time()-start
    return tst_matrix
    

def train(feature_matrix, sentiments, tst_matrix, tst_sent): 
    '''
    Train SVMs to predict sentiments with given feature words
    '''
    start = time.time()
    (n, d) = feature_matrix.shape
    w_A = np.zeros((1, d))
    w_P = np.ones((1, d)) / math.sqrt(L * d)
    s = np.ones((1,d))
    errTrain_P = []
    errTst_P = []
    errTrain_A = []
    errTst_A = []
    
    for t in range(T+1):
        idx = np.random.randint(n, size=B)
        X = feature_matrix[idx,:].todense()
        y = sentiments[idx]
        Eta = 1 / (L * (t+1))
        
        w_P = PEGASOS(w_P, X, y, Eta)
        (w_A, s) = AdaGrad(w_A, X, y, Eta, s)

        if t % 10 == 0:
            errTrain_P.append(predict(w_P, X, y))
            errTst_P.append(predict(w_P, tst_matrix, tst_sent))
            errTrain_A.append(predict(w_A, X, y))
            errTst_A.append(predict(w_A, tst_matrix, tst_sent))
            print t*B
    
    accuracy_P = 1 - predict(w_P, tst_matrix, tst_sent)
    accuracy_A = 1 - predict(w_A, tst_matrix, tst_sent)
    print "PEGASOS_accuracy = ", accuracy_P
    print "AdaGrad_accuracy = ", accuracy_A
    print "Time = ", time.time()-start
    plotCurves(errTrain_P, errTst_P, errTrain_A, errTst_A)


def PEGASOS(w, X, y, Eta):
    '''
    Train a SVM through PEGASOS
    '''
    Aplus = np.where(y * np.squeeze(np.asarray(X.dot(w.T))) < 1)
    g = y[Aplus] * X[Aplus]
    _w = (1 - Eta * L) * w + (Eta / B) * g
    norm = math.sqrt(L * _w.dot(_w.T))
    w = _w if 1 < norm else _w / norm
    return w


def AdaGrad(w, X, y, Eta, s):
    '''
    Train a SVM through AdaGrad
    '''
    Aplus = np.where(y * np.squeeze(np.asarray(X.dot(w.T))) < 1)
    grad = -(y[Aplus] * X[Aplus])/B + L*w
    G_inv= 1 / (np.sqrt(s) + epson)
    w -= Eta * np.multiply(G_inv, grad)
    s += np.square(grad)
    return (w, s)
        

def predict(w, pred_matrix, pred_sent):
    '''
    Predict the sentiments with given features and w, calculate the error
    '''
    n = len(pred_sent)
    pred = pred_matrix.dot(w.T)
    diff = (np.sign(pred).T - pred_sent) / 2
    error = diff.dot(diff.T) / n
    return error[0,0]


def plotCurves(errTrain_P, errTst_P, errTrain_A, errTst_A):
    '''
    Plot the learning curves
    '''
    maxY = 1
    maxX = len(errTst_A)
    xs = np.arange(maxX)
    plt.plot(xs, errTrain_P, 'r-')
    plt.hold(True)
    plt.plot(xs, errTst_P, 'b-')
    plt.hold(True)
    plt.plot(xs, errTrain_A, 'k-')
    plt.hold(True)
    plt.plot(xs, errTst_A, 'g-')
    plt.legend(['PEGASOS Training Error', 'PEGASOS Testing Error', 'AdaGrad Training Error', 'AdaGrad Testing Error'], loc = 'best')
    plt.title('PEGASOS (batch='+str(B)+', lambda='+str(L)+', iterations='+str(T*B)+')')
    plt.xlabel('Training iteration x1000')
    plt.ylabel('Error')
    plt.ylim((0,maxY))
    plt.xlim((0,maxX+1))
    plt.show()


def main():
    '''
    This is where the program starts
    '''
    print("Reading training data...")
#    (features, sentiments) = readCleanData('./trainingandtestdata/training.csv')# FOR TESTING!!!! 20000 entities
    (features, sentiments) = readCleanData('./trainingandtestdata/training.1600000.processed.noemoticon.csv')
    print("Reading complete \n")
    
    print("Reading testing data...")
    (tst_feat, tst_sent) = readCleanData('./trainingandtestdata/testdata.manual.2009.06.14.csv')
    print("Reading complete \n")
    
    print("Creating feature matrix...")
    (feature_matrix, word_bag) = geneFeatureMatrix(features)
    print("Feature matrix created \n")
    
    print("Creating testing matrix...")
    tst_matrix = geneTestingMatrix(tst_feat, word_bag)
    print("Testing matrix created \n")
    
    print("Training...")
    train(feature_matrix, sentiments, tst_matrix, tst_sent)


if __name__ == '__main__':
    main()