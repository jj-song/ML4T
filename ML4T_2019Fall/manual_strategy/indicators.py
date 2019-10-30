import pandas as pd
import numpy as np
from util import get_data, plot_data
import matplotlib.pyplot as plt
import datetime as dt
from manual_strategy.marketsimcode import get_symbols_prices, df_trades_transform, \
    compute_portvals, get_portfolio_stats

def get_rolling_mean(values, window):
    rolling_mean = values.rolling(window).mean()
    return rolling_mean

def get_rolling_std(values, window):
    rolling_std = values.rolling(window).std()
    return rolling_std

def get_bollinger_bands(rolling_mean, rolling_std):
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return upper_band, lower_band

def get_stochastic(values, window):
    rolling_min = values.rolling(window).min()
    rolling_max = values.rolling(window).max()
    k = ((values-rolling_min) / (rolling_max-rolling_min)) * 100

    return k


def main():
    sd = dt.datetime(2010,1,1)
    ed = dt.datetime(2011,12,31)
    symbol = 'JPM'
    window = 20

    symbols_prices_df = get_symbols_prices([symbol], sd, ed)
    symbols_prices_df = symbols_prices_df.fillna(method="ffill")
    symbols_prices_df = symbols_prices_df.fillna(method="bfill")
    prices = symbols_prices_df[symbol].to_frame()
    normalized_prices = prices / prices.iloc[0,]
    rolling_mean = get_rolling_mean(normalized_prices, window)
    rolling_std = get_rolling_std(normalized_prices, window)
    upper_band, lower_band = get_bollinger_bands(rolling_mean, rolling_std)
    stochastic = get_stochastic(normalized_prices, window)

    #plot simple moving average along with price
    plt.plot(normalized_prices, 'r', label="Normalized Prices")
    plt.plot(rolling_mean, 'g', label="SMA")
    plt.legend()
    plt.title("Normalized Price vs Simple Moving Average (SMA)")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.savefig('Normalized Price vs Simple Moving Average (SMA).png')
    plt.clf()

    #plot simple moving average along with price
    plt.plot(upper_band, 'b', label="Upper Band")
    plt.plot(lower_band, 'b', label="Lower Band")
    plt.plot(rolling_mean, 'g', label="SMA")
    plt.plot(normalized_prices, 'r', label="Normalized Prices")
    plt.legend()
    plt.title("Normalized Price with SMA by Bollinger Bands")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.savefig('Normalized Price with SMA by Bollinger Bands')
    plt.clf()

    #plot stochastic along with price
    #plt.plot(normalized_prices, 'r', label="Normalized Prices")
    plt.plot(stochastic, 'b', linewidth=.5, label="Stochastic")
    plt.legend()
    plt.title("Normalized Price with Stochastic Oscillator")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.savefig('Normalized Price with Stochastic Oscillator')
    plt.clf()

if __name__ == "__main__":
    main()
