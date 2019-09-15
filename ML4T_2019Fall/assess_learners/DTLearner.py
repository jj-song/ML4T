import DTLearner as dt
import numpy as np


class DTLearner(object):

    def __init__(self, leaf_size=1, verbose = False):
        self.leaf_size=leaf_size
        #pass # move along, these aren't the drones you're looking for

    def author(self):
        return 'jsong350' # replace tb34 with your Georgia Tech username

    def build_tree(self, dataX, dataY):
        #check if the leaf size is bigger than elements left. If so, create leaf and take mean of Y.
        if dataX.shape[0] <= self.leaf_size:
            return np.array([[-1, dataY.mean(), -1, -1]])
        #check if all of the dataY is the same. If so, create leaf and take the first Y value.
        elif np.all(dataY[:] == dataY[0]):
            return np.array([[-1, dataY[0], -1, -1]])
        #determine best feature i to split on
        else:
            features = {}
            for i in range(0, dataX.shape[1]):
                corr_matrix = np.corrcoef(dataX[:,i], dataY) #gets all combinations of correlations
                corr = corr_matrix[0,1] #gets correlation between data[:,i] and dataY
                abs_corr = abs(corr)
                features[i] = abs_corr

            print("these are the features with their correlation " + str(features))
            i = int(max(features, key=features.get)) #index of best feature to do initial split
            print("the feature used is " + str(i))
            #get the median value from the split column
            SplitVal = np.median(dataX[:,i])
            split_index_left = dataX[:,i] <= SplitVal
            split_index_right = dataX[:,i] > SplitVal

            #check if either side anything in it. If either side doensn't have anything, return mean of Y.
            if not np.any(split_index_left) or not np.any(split_index_right):
                return  np.array([[-1,dataY.mean(),-1,-1]])

            #create recursive left tree using data which had X values less than SplitVal
            lefttree = self.build_tree(dataX[dataX[:,i] <= SplitVal], dataY[dataX[:,i] <= SplitVal])
            #create recursive right tree using data which had X values greater than SplitVal
            righttree = self.build_tree(dataX[dataX[:,i] > SplitVal], dataY[dataX[:,i] > SplitVal])

            #recursively add new root after forming trees
            #factor, SplitVal, Left, Right
            root = np.array( [[i, SplitVal, 1, lefttree.shape[0] + 1]] ) #find new root

            #add left side of tree
            tree_structure = np.append(root, lefttree, axis=0)

            #add right side of tree
            tree_structure = np.append(tree_structure, righttree, axis=0)



            return tree_structure


    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.tree = self.build_tree(dataX, dataY)



    def query(self,testX):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        # loop through each of the test rows by each factor.
        testY = []
        for element in testX:
            count = 0

            while count != -1:
                factor = int(self.tree[count][0])
                SplitVal_or_Y = self.tree[count][1]
            # if the factor is -1, then stop.
                if factor == -1:
                    testY = np.append(testY, SplitVal_or_Y)
                    count = -1
                else:
                    if element[factor] <= SplitVal_or_Y:
                        count = count + int(self.tree[count][2])
                    elif element [factor] > SplitVal_or_Y:
                        count = count + int(self.tree[count][3])

        return testY



if __name__=="__main__":
    print("the secret clue is 'zzyzx'")
