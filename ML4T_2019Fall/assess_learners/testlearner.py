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
import math
import LinRegLearner as lrl
import sys
import DTLearner as dt

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

    # create a learner and train it
    learner = dt.DTLearner(leaf_size = 1, verbose = True) # create a LinRegLearner
    learner.addEvidence(trainX, trainY) # train it
    print(learner.author())

    # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    print("this is predY " + str(predY))
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(predY, y=trainY)
    print(f"corr: {c[0,1]}")

    # evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(predY, y=testY)
    print(f"corr: {c[0,1]}")
