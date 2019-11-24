import pandas as pd
import numpy as np
from util import get_data, plot_data
import matplotlib.pyplot as plt
import datetime as dt
from manual_strategy.marketsimcode import get_symbols_prices

def author():
    return 'jsong350'

def get_rolling_mean(values, window):
    rolling_mean = values.rolling(window).mean()
    return rolling_mean

def get_rolling_std(values, window):
    rolling_std = values.rolling(window).std()
    return rolling_std

def get_bollinger_bands(rolling_mean, rolling_std):
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    upper_band_one_std = rolling_mean + (rolling_std)
    lower_band_one_std = rolling_mean - (rolling_std)
    return upper_band, lower_band, upper_band_one_std, lower_band_one_std

def get_stochastic(values, window):
    rolling_min = values.rolling(window).min()
    rolling_max = values.rolling(window).max()
    k = ((values-rolling_min) / (rolling_max-rolling_min)) * 100
    return k

def get_all_indicators(sd, ed, symbol, window, plot):
    symbols_prices_df = get_symbols_prices([symbol], sd, ed)
    symbols_prices_df = symbols_prices_df.fillna(method="ffill")
    symbols_prices_df = symbols_prices_df.fillna(method="bfill")
    prices = symbols_prices_df[symbol].to_frame()
    normalized_prices = prices / prices.iloc[0,]
    rolling_mean = get_rolling_mean(normalized_prices, window)
    rolling_std = get_rolling_std(normalized_prices, window)
    upper_band, lower_band, upper_band_one_std, lower_band_one_std = get_bollinger_bands(rolling_mean, rolling_std)
    k = get_stochastic(normalized_prices, window)

    return rolling_mean, upper_band, lower_band, upper_band_one_std, lower_band_one_std, rolling_std, k

def main():
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
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
    joined = normalized_prices.join(rolling_mean, lsuffix = "Normalized Price", rsuffix = "SMA")
    joined.columns = ["Normalized Price", "SMA"]
    fig = joined.plot(title = "Normalized Price with SMA Overlay", fontsize=12, lw=1)
    fig.set_xlabel("Date")
    fig.set_ylabel("Price")
    plt.savefig("Normalized Price with SMA Overlay")
    plt.clf()

    #plot SMA and Normalized Price and Bollinger Bands
    joined = normalized_prices.join(rolling_mean, lsuffix = "np", rsuffix = "sma").join(upper_band, lsuffix = "", rsuffix = "ub").join(lower_band, lsuffix="", rsuffix="lb")
    joined.columns = ["Normalized Price", "SMA", "Upper Band", "Lower Band"]
    fig = joined.plot(title = "Normalized Price with Bollinger Bands including SMA", fontsize=12, lw=1)
    fig.set_xlabel("Date")
    fig.set_ylabel("Price")
    plt.savefig("Normalized Price with Bollinger Bands including SMA")
    plt.clf()

    #create subplot with normalized price and stochastic oscillator
    fig, axs = plt.subplots(2, sharey='row')
    fig.suptitle("Normalized Price with Stochastic Oscillator Overlay")
    axs[0].plot(normalized_prices, 'tab:orange')
    axs[1].plot(stochastic, 'tab:blue')
    axs[0].set(ylabel="Price")
    axs[1].set(ylabel="Reading")
    plt.xticks(rotation=30)
    plt.setp(axs[0].xaxis.get_majorticklabels(), visible=False)
    plt.savefig("Normalized Price with Stochastic Oscillator")
    plt.clf()


if __name__ == "__main__":
    main()
