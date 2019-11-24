"""  		   	  			  	 		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
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
  		   	  			  	 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		   	  			  	 		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		   	  			  	 		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		   	  			  	 		  		  		    	 		 		   		 		  
"""  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
import datetime as dt  		   	  			  	 		  		  		    	 		 		   		 		  
import pandas as pd  		   	  			  	 		  		  		    	 		 		   		 		  
import util as ut  		   	  			  	 		  		  		    	 		 		   		 		  
import random
import strategy_learner.BagLearner as bl
from strategy_learner.indicators import *
import strategy_learner.RTLearner as rt
  		   	  			  	 		  		  		    	 		 		   		 		  
class StrategyLearner(object):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # constructor  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose = False, impact=0.0):  		   	  			  	 		  		  		    	 		 		   		 		  
        self.verbose = verbose  		   	  			  	 		  		  		    	 		 		   		 		  
        self.impact = impact
        self.N = 10
        #self.learner = bl.BagLearner(kwargs = {"leaf_size":5, "verbose": False}, bags = 20)
        self.learner = bl.BagLearner(learner = rt.RTLearner, kwargs ={"leaf_size":5}, bags = 25, boost = False, verbose = False)

    def get_x_train_data(self, prices, sd, ed, window, symbol):
        rolling_mean, upper_band, lower_band, upper_band_one_std, lower_band_one_std, \
        rolling_std, k = get_all_indicators(sd, ed, symbol, window, False)

        x_train_data = np.zeros((len(prices) - self.N, 9))

        for index in range(0, len(prices) - self.N,9):
            x_train_data[index][0] = prices.iloc[index]
            x_train_data[index][1] = rolling_mean.iloc[index]
            x_train_data[index][2] = rolling_std.iloc[index]
            x_train_data[index][3] = upper_band_one_std.iloc[index]
            x_train_data[index][4] = lower_band_one_std.iloc[index]
            x_train_data[index][5] = upper_band.iloc[index]
            x_train_data[index][6] = lower_band.iloc[index]
            x_train_data[index][7] = k.iloc[index]
            x_train_data[index][8] = prices.iloc[index + self.N]
        return x_train_data

    def get_x_test_data(self, prices, sd, ed, window, symbol):
        rolling_mean, upper_band, lower_band, upper_band_one_std, lower_band_one_std, \
        rolling_std, k = get_all_indicators(sd, ed, symbol, window, False)

        x_test_data = np.zeros((len(prices) - self.N, 9))

        for index in range(0, len(prices) - self.N,9):
            x_test_data[index][1] = rolling_mean.iloc[index]
            x_test_data[index][2] = rolling_std.iloc[index]
            x_test_data[index][3] = upper_band_one_std.iloc[index]
            x_test_data[index][4] = lower_band_one_std.iloc[index]
            x_test_data[index][5] = upper_band.iloc[index]
            x_test_data[index][6] = lower_band.iloc[index]
        return x_test_data
  		   	  			  	 		  		  		    	 		 		   		 		  
    # this method should create a QLearner, and train it for trading  		   	  			  	 		  		  		    	 		 		   		 		  
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):  		   	  			  	 		  		  		    	 		 		   		 		  

        # example usage of the old backward compatible util function  		   	  			  	 		  		  		    	 		 		   		 		  
        syms=[symbol]  		   	  			  	 		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed)  		   	  			  	 		  		  		    	 		 		   		 		  
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		   	  			  	 		  		  		    	 		 		   		 		  
        prices = prices_all[syms]  # only portfolio symbols  		   	  			  	 		  		  		    	 		 		   		 		  
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later  		   	  			  	 		  		  		    	 		 		   		 		  
        if self.verbose: print(prices)

        # add your code to do learning here
        x_train_data = self.get_x_train_data(prices, sd, ed, syms)
        x_train_data = x_train_data[:, 1:-1]

        y_train_data = []
        for index in range(0, x_train_data.shape[0]):
            if x_train_data[index, -1] / x_train_data[index, 0] > 1.02 + self.impact:
                y_train_data.append(1000)
            elif x_train_data[index, -1] / x_train_data[index, 0] < 0.98-self.impact:
                y_train_data.append(-1000)
            else:
                y_train_data.append(0)

        y_train_data = np.array(y_train_data)

        self.learner.addEvidence(x_train_data, y_train_data)
  		   	  			  	 		  		  		    	 		 		   		 		  
        # # example use with new colname
        # volume_all = ut.get_data(syms, dates, colname = "Volume")  # automatically adds SPY
        # volume = volume_all[syms]  # only portfolio symbols
        # volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        # if self.verbose: print(volume)
  		   	  			  	 		  		  		    	 		 		   		 		  
    # this method should use the existing policy and test it against new data  		   	  			  	 		  		  		    	 		 		   		 		  
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print(prices)

        x_train_data = self.get_x_train_data(prices_SPY, sd, ed, syms)
        query = self.learner.query(x_train_data)
        trades = pd.DataFrame(0, columns = prices.columns, index= prices.index)

        share = 0
        for index in range(0, len(prices) - self.N):
            if query[index] == 1000:
                if share ==0:
                    share = 1000
                    trades.iloc[index, 0] = 1000
                elif share == -1000:
                    share = 1000
                    trades.iloc[index, 0] = 2000
            if query[index] == -1000:
                if share ==0:
                    share = -1000
                    trades.iloc[index, 0] = -1000
                elif share ==1000:
                    share = -1000
                    trades.iloc[index, 0] = -2000

  		   	  			  	 		  		  		    	 		 		   		 		  
        # here we build a fake set of trades  		   	  			  	 		  		  		    	 		 		   		 		  
        # your code should return the same sort of data  		   	  			  	 		  		  		    	 		 		   		 		  
        # dates = pd.date_range(sd, ed)
        # prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        # trades = prices_all[[symbol,]]  # only portfolio symbols
        # trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        # trades.values[:,:] = 0 # set them all to nothing
        # trades.values[0,:] = 1000 # add a BUY at the start
        # trades.values[40,:] = -1000 # add a SELL
        # trades.values[41,:] = 1000 # add a BUY
        # trades.values[60,:] = -2000 # go short from long
        # trades.values[61,:] = 2000 # go long from short
        # trades.values[-1,:] = -1000 #exit on the last day
        if self.verbose: print(type(trades)) # it better be a DataFrame!  		   	  			  	 		  		  		    	 		 		   		 		  
        if self.verbose: print(trades)  		   	  			  	 		  		  		    	 		 		   		 		  
        if self.verbose: print(prices_all)  		   	  			  	 		  		  		    	 		 		   		 		  
        return trades  		   	  			  	 		  		  		    	 		 		   		 		  

if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    print("One does not simply think up a strategy")  		   	  			  	 		  		  		    	 		 		   		 		  
