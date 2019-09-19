import numpy as np
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import LinRegLearner as lrl

class BagLearner(object):

    def __init__(self, learner=None, kwargs={}, bags=25, boost=False, verbose = False):
        #Check for DTLearner and RTLearner. If either, designate leaf_size.
        #if learner in [dt.DTLearner, rt.RTLearner]:

        self.learners = []
        for i in range(0, bags):
            self.learners.append(learner(**kwargs))

    def author(self):
        return 'jsong350' # replace tb34 with your Georgia Tech username


    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        #total number of rows
        num_of_rows = int(dataX.shape[0])

        for learner in self.learners:
            random_select = np.random.choice(num_of_rows, num_of_rows, replace=True)
            train_data = dataX[random_select]
            test_data = dataY[random_select]
            #add data to learners
            learner.addEvidence(train_data, test_data)


    def query(self,test):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        # loop through each of the test rows by each factor.
        aggregate = np.array([learner.query(test) for learner in self.learners])
        return np.mean(aggregate, axis=0)



if __name__=="__main__":
    print("the secret clue is 'zzyzx'")
