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
from indicators import get_all_factors, adjust_for_nan
from marketsimcode import calculate_reward, translate_action, discretize, combine_indicators, prepare_dataframes

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.ql = ql.QLearner(num_states=10000, num_actions=3, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False)


    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,12,31), \
        sv = 10000):
        anchor_sd = sd
        window = 20

        # add your code to do learning here
        # use sd for calculating indicators
        sd = adjust_for_nan(sd, window)

        # example usage of the old backward compatible util function
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        if self.verbose: print(prices)

        normalized_prices, rolling_mean, upper_band, lower_band, rolling_std, k = \
            get_all_factors(sd, ed, syms, window, False)
        normalized_prices = normalized_prices.loc[anchor_sd:]
        all_indicators = combine_indicators(rolling_mean, upper_band, lower_band, k, anchor_sd)

        d_indicators = discretize(all_indicators)
        reward_multiplier = calculate_reward(normalized_prices)
        symbol_df, order_num_df, trades_df = prepare_dataframes(normalized_prices, symbol)
        count = 0
        df = pd.concat([symbol_df, order_num_df, trades_df], axis=1)
        df.columns = ['Symbol', 'Order', 'Shares']

        while count < 200:
            if count > 15 and df.equals(df_c):
                break
            df_c = df.copy()
            count +=1
            net_holdings = 0
            #print("this is " + str(count) + " time")
            for datetime, norm_price in normalized_prices.iterrows():
                reward = net_holdings * reward_multiplier.loc[datetime] * (1 - self.impact)
                result = self.ql.query(int(float(d_indicators.loc[datetime]['states'])), reward)
                action = translate_action(result)
                if (action == 'BUY') and (net_holdings < 1000):
                    trades_df.loc[datetime]['Order'] = action
                    if net_holdings == 0:
                        order_num_df.loc[datetime]['Shares'] = 1000
                        net_holdings += 1000
                    elif net_holdings == -1000:
                        order_num_df.loc[datetime]['Shares'] = 2000
                        net_holdings += 2000
                elif (action == 'SELL') and (net_holdings > -1000):
                    trades_df.loc[datetime]['Order'] = action
                    if net_holdings == 0:
                        order_num_df.loc[datetime]['Shares'] = -1000
                        net_holdings -= 1000
                    elif net_holdings == 1000:
                        order_num_df.loc[datetime]['Shares'] = -2000
                        net_holdings -= 2000
                else:
                    pass

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2010,1,1), \
        ed=dt.datetime(2011,12,31), \
        sv = 10000):
        anchor_sd = sd
        window = 20

        sd = adjust_for_nan(sd, window)

        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        if self.verbose: print(prices)

        normalized_prices, rolling_mean, upper_band, lower_band, rolling_std, k = \
            get_all_factors(sd, ed, syms, window, False)
        normalized_prices = normalized_prices.loc[anchor_sd:]
        all_indicators = combine_indicators(rolling_mean, upper_band, lower_band, k, anchor_sd)



        symbol_df, order_num_df, trades_df = prepare_dataframes(normalized_prices, symbol)
        net_holdings = 0
        d_indicators = discretize(all_indicators)
        for datetime, norm_price in normalized_prices.iterrows():
            result = self.ql.querysetstate(int(float(d_indicators.loc[datetime]['states'])))
            action = translate_action(result)
            if (action == 'BUY') and (net_holdings < 1000):
                trades_df.loc[datetime]['Order'] = action
                if net_holdings == 0:
                    order_num_df.loc[datetime]['Shares'] = 1000
                    net_holdings += 1000
                elif net_holdings == -1000:
                    order_num_df.loc[datetime]['Shares'] = 2000
                    net_holdings += 2000
            elif (action == 'SELL') and (net_holdings > -1000):
                trades_df.loc[datetime]['Order'] = action
                if net_holdings == 0:
                    order_num_df.loc[datetime]['Shares'] = -1000
                    net_holdings -= 1000
                elif net_holdings == 1000:
                    order_num_df.loc[datetime]['Shares'] = -2000
                    net_holdings -= 2000
            else:
                pass

        df = order_num_df
        df.columns = ['Shares']

        return df

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

    def author(self):
        return 'jsong350'

if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    print("One does not simply think up a strategy")  		   	  			  	 		  		  		    	 		 		   		 		  
