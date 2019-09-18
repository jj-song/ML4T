"""
Test a learner.  (c) 2015 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import math
import LinRegLearner as lrl
import InsaneLearner as it
import sys
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import pandas as pd

if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    #data = np.array([list(map(float,s.strip().split(','))) for s in inf.readlines()])
    #print(inf.readlines())
    data = np.array([list(map(float,s.strip().split(',')[1:])) for s in inf.readlines()[1:]])

    # compute how much of the data is training and testing
    train_rows = int(0.6* data.shape[0])  #takes 60% of rows for train set
    test_rows = data.shape[0] - train_rows #takes remaining 40% for testing

    # separate out training and testing data
    #trainX = data[:train_rows,0:-1]  #take all of the training rows and get all columns except
    trainX = data[:train_rows,1:-1]  #take all of the training rows and get all columns except last	column
    trainY = data[:train_rows,-1] #take all of training rows and get only the last column
    #testX = data[train_rows:,0:-1] #take the remaining rows and get all columns except last
    testX = data[train_rows:,1:-1] #take the remaining rows and get all columns except last
    testY = data[train_rows:,-1] #take the test rows and get only the last column

    print(f"{testX.shape}") #will show (number of rows, number of column) for test x
    print(f"{testY.shape}") #will show (number of rows, number of column) for test y

    ############### DTLearner ###############
    # create decision tree learner and train it
    #learner = dt.DTLearner(leaf_size = 50, verbose = True) # create a LinRegLearner
    #learner.addEvidence(trainX, trainY) # train it
    #print(learner.author())
    #########################################

    ############### RTLearner ###############
    # create random tree learner and train it
    #learner = rt.RTLearner(leaf_size = 50, verbose = True) # create a LinRegLearner
    #learner.addEvidence(trainX, trainY) # train it
    #print(learner.author())
    #########################################

    ############### BagLearner ##############
    # create bag learner and train it
    #learner = bl.BagLearner(learner = rt.RTLearner, leaf_size=20, bags = 20, boost = False, verbose = False)
    #learner.addEvidence(trainX, trainY) # train it
    #print(learner.author())
    #########################################

    ############### InsaneLearner ###########
    # create insane learner and train it
    #learner = it.InsaneLearner(verbose = False) # constructor
    #learner.addEvidence(trainX, trainY) # train it
    #print(learner.author())
    #########################################


    # Question 1: Does overfitting occur with respect to leaf_size? Use the dataset istanbul.csv with DTLearner. For which values of leaf_size does overfitting occur? Use RMSE as your metric for assessing overfitting. Support your assertion with graphs/charts. (Don't use bagging).
    '''
    in_sample = np.zeros((100, 1))
    for leaf in range(1, 101):
        learner = dt.DTLearner(leaf_size = leaf, verbose = True)
        learner.addEvidence(trainX, trainY)
        predY = learner.query(trainX) # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        in_sample[leaf-1] = rmse
        print("Leaf")
        print(leaf)
        print("RMSE")
        print(rmse)

    out_sample = np.zeros((100, 1))
    for leaf in range(1, 101):
        learner = dt.DTLearner(leaf_size = leaf, verbose = True)
        learner.addEvidence(trainX, trainY)
        predY = learner.query(testX) # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        out_sample[leaf-1] = rmse
        print("Leaf")
        print(leaf)
        print("RMSE")
        print(rmse)

    plt.plot(out_sample, label="Out of Sample")
    plt.plot(in_sample, label="In Sample")
    plt.legend()
    plt.title("Leaf vs RMSE Without Bagging")
    plt.xlim([-10, 100])
    plt.xlabel("Leaf")
    plt.ylabel("RMSE")
    plt.savefig('leaf_vs_rmse_dtlearn_no_bag.png')
    plt.clf()
    '''


    # Question 2: Can bagging reduce or eliminate overfitting with respect to leaf_size? Again use the dataset istanbul.csv with DTLearner. To investigate this choose a fixed number of bags to use and vary leaf_size to evaluate. Provide charts to validate your conclusions. Use RMSE as your metric.
    '''
    in_sample = np.zeros((50, 1))
    for leaf in range(1, 51):
        learner = bl.BagLearner(learner = dt.DTLearner, kwargs ={"leaf_size":leaf}, bags = 25, boost = False, verbose = False)
        learner.addEvidence(trainX, trainY) # train it
        predY = learner.query(trainX) # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        in_sample[leaf-1] = rmse
        print("Leaf")
        print(leaf)
        print("RMSE")
        print(rmse)

    out_sample = np.zeros((50, 1))
    for leaf in range(1, 51):
        learner = bl.BagLearner(learner = dt.DTLearner, kwargs ={"leaf_size":leaf}, bags = 25, boost = False, verbose = False)
        learner.addEvidence(trainX, trainY) # train it
        predY = learner.query(testX) # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        out_sample[leaf-1] = rmse
        print("Leaf")
        print(leaf)
        print("RMSE")
        print(rmse)

    plt.plot(out_sample, label="Out of Sample")
    plt.plot(in_sample, label="In Sample")
    plt.legend()
    plt.title("Leaf vs RMSE With Bagging")
    plt.xlim([-.0010, .0070])
    plt.xlim([0, 55])
    plt.xlabel("Leaf")
    plt.ylabel("RMSE")
    plt.savefig('leaf_vs_rmse_dtlearn_yes_bag.png')
    plt.clf()
    '''


    # Question 3: Quantitatively compare "classic" decision trees (DTLearner) versus random trees (RTLearner). In which ways is one method better than the other? Provide at least two quantitative measures. Important, using two similar measures that illustrate the same broader metric does not count as two. (For example, do not use two measures for accuracy.) Note for this part of the report you must conduct new experiments, don't use the results of the experiments above for this(RMSE is not allowed as a new experiment).

    for exp in range(0, 4):
        dt_train_t = np.zeros((4,50))
        dt_query_t = np.zeros((4,50))
        dt_size = np.zeros(50,)
        rt_size = np.zeros(50,)
        for leaf in range(1, 51):
            startTime = time.time()
            learner = dt.DTLearner(leaf_size = leaf, verbose = True)
            learner.addEvidence(trainX, trainY)
            dt_train_t[exp][leaf-1] = time.time()-startTime
            dt_train_avg = np.mean(dt_train_t, axis=0)
            print(dt_train_avg)

            startTime = time.time()
            predY = learner.query(testX) # get the predictions
            dt_query_t[exp][leaf-1] = time.time()-startTime
            dt_query_avg = np.mean(dt_query_t, axis=0)

            dt_size[leaf-1] = learner.tree.shape[0]


        rt_train_t = np.zeros((4, 50))
        rt_query_t = np.zeros((4, 50))
        for leaf in range(1, 51):
            startTime = time.time()
            learner = rt.RTLearner(leaf_size = leaf, verbose = True)
            learner.addEvidence(trainX, trainY)
            rt_train_t[exp][leaf-1] = time.time()-startTime
            rt_train_avg = np.mean(rt_train_t, axis=0)

            startTime = time.time()
            predY = learner.query(testX) # get the predictions
            rt_query_t[exp][leaf-1] = time.time()-startTime
            rt_query_avg = np.mean(rt_query_t, axis=0)

            rt_size[leaf-1] = learner.tree.shape[0]


    plt.plot(dt_train_avg, lw=1, label="DTLearner Train Time")
    plt.plot(rt_train_avg, lw=1, label="RTLearner Train Time")
    plt.plot(dt_query_avg, lw=1, label="DTLearner Query Time")
    plt.plot(rt_query_avg, lw=1, label="RTLearner Query Time")
    plt.legend()
    plt.title("Time vs Leaf - Compare DT and RT Learners' Train and Query Times")
    ##plt.xlim([-10, 550])
    plt.xlabel("Leaf")
    plt.ylabel("Seconds")
    plt.savefig('leaf_vs_query_train_time_dt_vs_rt.png')
    plt.clf()


    plt.plot(dt_size, lw=1, label="DTLearner Size")
    plt.plot(rt_size, lw=1, label="RTLearnerSize")
    plt.legend()
    plt.title("Nodes vs Leaf - Compare DT and RT Learners' Tree Sizes")
    ##plt.xlim([-10, 550])
    plt.xlabel("Leaf")
    plt.ylabel("Nodes")
    plt.savefig('leaf_vs_nodes_dt_vs_rt.png')
    plt.clf()
