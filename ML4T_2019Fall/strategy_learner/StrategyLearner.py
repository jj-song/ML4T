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
import numpy as np
import QLearner as ql
from indicators import get_all_indicators
from marketsimcode import get_daily_returns, discretize

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.ql = ql.QLearner(num_states=10000, num_actions=3, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False)

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):

        N=30
        K=2*N
        sd_original = sd
        window = 20

        # add your code to do learning here
        sd = sd - dt.timedelta(K)

        # example usage of the old backward compatible util function
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print(prices)

        prices_normalized, rolling_mean, upper_band, lower_band, rolling_std, k = \
            get_all_indicators(sd, ed, syms, window, False)

        indicators = pd.concat([rolling_mean, upper_band, lower_band, k], axis=1)
        indicators = indicators.loc[sd_original:]
        indicators.columns = ['rolling_mean', 'upper_band', 'lower_band', 'k']
        prices_normalized = prices_normalized.loc[sd_original:]

        daily_price_change = get_daily_returns(prices_normalized)

        indicators = discretize(indicators)

        initial_state = indicators.iloc[0]['state']

        self.ql.querysetstate((int(float(initial_state))))

        orders = pd.DataFrame(0, index = prices_normalized.index, columns = ['Shares'])
        buy_sell = pd.DataFrame('BUY', index=prices_normalized.index, columns=['Order'])
        symbol_df = pd.DataFrame(symbol, index=prices_normalized.index, columns=['Symbol'])

        df_trades = pd.concat([symbol_df, buy_sell, orders], axis=1)
        df_trades.columns = ['Symbol', 'Order', 'Shares']

        df_trades_copy = df_trades.copy()

        i = 0

        while i < 250:
            i +=1
            reward = 0
            total_holdings = 0
            print("this is " + str(i) + " time")
            df_trades_copy = df_trades.copy()

            for index, row in prices_normalized.iterrows():
                reward = total_holdings * daily_price_change.loc[index] * (1 - self.impact)
                a = self.ql.query(int(float(indicators.loc[index]['state'])), reward)
                if(a ==1) and (total_holdings < 1000):
                    buy_sell.loc[index]['Order'] = 'BUY'
                    if total_holdings ==0:
                        orders.loc[index]['Shares'] = 1000
                        total_holdings += 1000
                    else:
                        orders.loc[index]['Shares'] = 2000
                        total_holdings += 2000
                elif (a ==2) and (total_holdings > -1000):
                    buy_sell.loc[index]['Order'] = 'SELL'
                    if total_holdings == 0:
                        orders.loc[index]['Shares'] = -1000
                        total_holdings = total_holdings -1000
                    else:
                        orders.loc[index]['Shares'] = -2000
                        total_holdings = total_holdings - 2000

            df_trades = pd.concat([symbol_df, buy_sell, orders], axis=1)
            df_trades.columns = ['Symbol', 'Order', 'Shares']

        print(df_trades)


    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        N = 30
        K = N+30
        sd_original = sd
        window = 20

        sd = sd - dt.timedelta(K)

        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print(prices)

        prices_normalized, rolling_mean, upper_band, lower_band, rolling_std, k = \
            get_all_indicators(sd, ed, syms, window, False)

        indicators = pd.concat([rolling_mean, upper_band, lower_band, k], axis=1)
        indicators = indicators.loc[sd_original:]
        indicators.columns = ['rolling_mean', 'upper_band', 'lower_band', 'k']
        prices_normalized = prices_normalized.loc[sd_original:]

        daily_price_change = get_daily_returns(prices_normalized)

        indicators = discretize(indicators)

        initial_state = indicators.iloc[0]['state']

        self.ql.querysetstate((int(float(initial_state))))

        orders = pd.DataFrame(0, index = prices_normalized.index, columns = ['Shares'])
        buy_sell = pd.DataFrame('BUY', index=prices_normalized.index, columns=['Order'])
        symbol_df = pd.DataFrame(symbol, index=prices_normalized.index, columns=['Symbol'])

        reward = 0
        total_holdings = 0
        for index, row in prices_normalized.iterrows():
            reward = total_holdings * daily_price_change.loc[index]

            a = self.ql.querysetstate(int(float(indicators.loc[index]['state'])))
            if (a == 1) and (total_holdings < 1000):
                buy_sell.loc[index]['Order'] = 'BUY'
                if total_holdings == 0:
                    orders.loc[index]['Shares'] = 1000
                    total_holdings += 1000
                else:
                    orders.loc[index]['Shares'] = 2000
                    total_holdings += 2000
            elif (a == 2) and (total_holdings > -1000):
                buy_sell.loc[index]['Order'] = 'SELL'
                if total_holdings == 0:
                    orders.loc[index]['Shares'] = -1000
                    total_holdings = total_holdings - 1000
                else:
                    orders.loc[index]['Shares'] = -2000
                    total_holdings = total_holdings - 2000

        df_trades = pd.concat([symbol_df, buy_sell, orders], axis=1)
        df_trades.columns = ['Symbol', 'Order', 'Shares']

        df_trades = df_trades.drop('Symbol', axis=1)
        df_trades = df_trades.drop('Order', axis=1)

        return df_trades

        # # here we build a fake set of trades
        # # your code should return the same sort of data
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
        # if self.verbose: print(type(trades)) # it better be a DataFrame!
        # if self.verbose: print(trades)
        # if self.verbose: print(prices_all)
        # return trades

if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    print("One does not simply think up a strategy")  		   	  			  	 		  		  		    	 		 		   		 		  
