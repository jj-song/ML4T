"""MC2-P1: Market simulator.

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

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

#pd.set_option('display.max_columns', None)

def author():
    return 'jsong350'

def get_symbols(df):
    symbols = df.Symbol.unique().tolist()
    return symbols

def get_symbols_prices(symbols, start_date, end_date):
    symbols_prices_df = get_data(symbols, pd.date_range(start_date, end_date))
    return symbols_prices_df

def initialize_symbols_to_prices_df(df, symbols, start_val):
    for symbol in symbols:
        df[symbol + ' Shares'] = pd.Series(0, index=df.index)
        df['Portfolio Value'] = pd.Series(start_val, index=df.index)
        df['Cash'] = pd.Series(start_val, index=df.index)
    return df

def combine_indicators(rolling_mean, upper_band, lower_band, k, anchor_sd):
    all_indicators = pd.concat([rolling_mean, upper_band, lower_band, k], axis=1)
    all_indicators = all_indicators.loc[anchor_sd:]
    all_indicators.columns = ['rolling_mean', 'upper_band', 'lower_band', 'k']
    return all_indicators

def discretize(all_indicators):

    bin_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    rm_min = all_indicators['rolling_mean'].min()
    rm_max = all_indicators['rolling_mean'].max()
    rm_inc = (rm_max - rm_min) / 10
    bins_rm = [rm_min - rm_min / 10, rm_min + rm_inc, rm_min + rm_inc * 2, rm_min + rm_inc * 3, rm_min + rm_inc * 4,
               rm_min + rm_inc * 5, rm_min + rm_inc * 6, rm_min + rm_inc * 7, rm_min + rm_inc * 8, rm_min + rm_inc * 9,
               rm_max + rm_max / 10]
    all_indicators['rm'] = pd.cut(all_indicators['rolling_mean'], bins_rm, labels=bin_num)


    ub_min = all_indicators['upper_band'].min()
    ub_max = all_indicators['upper_band'].max()
    ub_inc = (ub_max - ub_min) / 10
    bins_ub = [ub_min - ub_min / 10, ub_min + ub_inc, ub_min + ub_inc * 2, ub_min + ub_inc * 3, ub_min + ub_inc * 4,
               ub_min + ub_inc * 5, ub_min + ub_inc * 6, ub_min + ub_inc * 7, ub_min + ub_inc * 8, ub_min + ub_inc * 9,
               ub_max + ub_max / 10]
    all_indicators['ub'] = pd.cut(all_indicators['upper_band'], bins_ub, labels=bin_num)

    lb_min = all_indicators['lower_band'].min()
    lb_max = all_indicators['lower_band'].max()
    lb_inc = (lb_max - lb_min) / 10
    bins_lb = [lb_min - lb_min / 10, lb_min + lb_inc, lb_min + lb_inc * 2, lb_min + lb_inc * 3, lb_min + lb_inc * 4,
               lb_min + lb_inc * 5, lb_min + lb_inc * 6, lb_min + lb_inc * 7, lb_min + lb_inc * 8, lb_min + lb_inc * 9,
               lb_max + lb_max / 10]
    all_indicators['lb'] = pd.cut(all_indicators['lower_band'], bins_lb, labels=bin_num)

    bins_k = [-0.01, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.1]
    all_indicators['stoc'] = pd.cut(all_indicators['k'], bins_k, labels=bin_num)

    all_indicators['states'] = all_indicators['rm'].astype(str) \
                          + all_indicators['ub'].astype(str) \
                          + all_indicators['lb'].astype(str) \
                          + all_indicators['stoc'].astype(str)


    return all_indicators

def calculate_reward(normalized_prices):
    reward = normalized_prices
    reward[1:] = (normalized_prices[1:] / normalized_prices[:-1].values) - 1
    return reward

def prepare_dataframes(normalized_prices, symbol):
    symbol_df = pd.DataFrame(symbol, index=normalized_prices.index, columns=['Symbol'])
    order_num_df = pd.DataFrame(0, index=normalized_prices.index, columns=['Shares'])
    trades_df = pd.DataFrame('', index=normalized_prices.index, columns=['Order'])

    return symbol_df, order_num_df, trades_df

def initalize_data(orders, start_val):
    #begin
    orders_df = orders
    start_date = min(orders_df.index)
    end_date = max(orders_df.index)
    symbols = get_symbols(orders_df)
    symbols_prices_df = get_symbols_prices(symbols, start_date, end_date)

    #create initial df using data points gathered above
    initialize_symbols_to_prices_df(symbols_prices_df, symbols, start_val)
    #print(symbols_prices_df)
    return orders_df, symbols_prices_df, symbols, start_date, end_date

def compute_buy_on_share(symbols_prices_df, index, row, symbol):
    symbols_prices_df.loc[index:,symbol + ' Shares'] = symbols_prices_df.loc[index:,symbol + ' Shares'] + row['Shares']

def compute_buy_on_cash(symbols_prices_df, index, row, symbol, impact, commission):
    symbols_prices_df.loc[index:,'Cash'] -= symbols_prices_df.loc[index,symbol] * row['Shares'] * (1 + impact) + commission

def compute_sell_on_share(symbols_prices_df, index, row, symbol):
    symbols_prices_df.loc[index:,symbol + ' Shares'] = symbols_prices_df.loc[index:,symbol + ' Shares'] - row['Shares']

def compute_sell_on_cash(symbols_prices_df, index, row, symbol, impact, commission):
    symbols_prices_df.loc[index:,'Cash'] += symbols_prices_df.loc[index,symbol] * row['Shares'] * (1 - impact) - commission

def get_portfolio_stats(pv):
    sf = 252
    daily_change = pv/pv.shift(1) - 1
    cum_ret = pv[-1]/pv[0] - 1
    avg_daily_ret = daily_change.mean()
    std_daily_ret = daily_change.std()
    sharpe_ratio = (np.sqrt(sf)*(avg_daily_ret))/std_daily_ret

    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio

#You need to run this otherwise it won't be in the correct orders format accepted by compute_portvals.
def df_trades_transform(df_trades, symbol):
    symbols = []
    orders = []
    shares = []

    for index in range(len(df_trades.index)):
        symbols.append(symbol)
        if df_trades['orders'][index] > 0:
            orders.append('BUY')
            shares.append(df_trades['orders'][index])
        elif df_trades['orders'][index] < 0:
            orders.append('SELL')
            shares.append(-df_trades['orders'][index])

    df_symbol = pd.DataFrame(data = symbols, index=df_trades.index, columns = ['Symbol'])
    df_order = pd.DataFrame(data=orders, index=df_trades.index, columns=['Order'])
    df_share = pd.DataFrame(data=shares, index=df_trades.index, columns=['Shares'])

    df_result = df_symbol.join(df_order).join(df_share)
    return df_result

def translate_action(result):
    if result == 1:
        return 'BUY'
    elif result == 2:
        return 'SELL'

def normalize(portval):
    return portval / portval.ix[0,:]

def compute_portvals(orders, start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    #get df of orders (prices not included)
    orders_df, symbols_prices_df, symbols, start_date, end_date = initalize_data(orders, start_val)

    for index, row in orders_df.iterrows():
        symbol = row['Symbol'] #e.g. GOOG
        order = row ['Order']
        if order == 'BUY':
            compute_buy_on_share(symbols_prices_df, index, row, symbol)
            compute_buy_on_cash(symbols_prices_df, index, row, symbol, impact, commission)
        if order == 'SELL':
            compute_sell_on_share(symbols_prices_df, index, row, symbol)
            compute_sell_on_cash(symbols_prices_df, index, row, symbol, impact, commission)

    for index, row in symbols_prices_df.iterrows():
        cash_value = 0 #this is cash equivalent of shares
        for symbol in symbols:
            cash_value += symbols_prices_df.loc[index, symbol + ' Shares'] * row[symbol]
            symbols_prices_df.loc[index, 'Portfolio Value'] = symbols_prices_df.loc[index, 'Cash'] + cash_value

    return symbols_prices_df.loc[:, 'Portfolio Value']

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-short.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    orders_df, symbols_prices_df, symbols, start_date, end_date = initalize_data(orders = of, start_val = sv)

    #get stats for fund
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

    #get df for SPY, then stats for the spx_prices_df
    spx_prices_df = get_symbols_prices(['$SPX'], start_date, end_date).loc[:, '$SPX']
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = get_portfolio_stats(spx_prices_df)

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")

if __name__ == "__main__":
    test_code()
