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
    upper_band = rolling_mean + rolling_std * 2
    lower_band = rolling_mean - rolling_std * 2
    return upper_band, lower_band

def get_momentum(values, window):
    test = values.shift(window)
    momentum = values/values.shift(window-1) - 1
    return momentum

def get_stochastic(values, window):
    rolling_min = values.rolling(window).min()
    rolling_max = values.rolling(window).max()
    k = ((values-rolling_min) / (rolling_max-rolling_min)) * 100

    return k


def main():
    sd = dt.datetime(2010,1,1)
    ed = dt.datetime(2011,12,31)


    #df_trades with optimal strategy
    df_trades = ms.testPolicy('JPM', sd, ed, 100000)
    df_trades_transformed = df_trades_transform(df_trades)
    portvals = compute_portvals(df_trades_transformed, start_val=100000, commission=0, impact=0)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

    #df_trades with benchmark (buy once, hold, sell at end)
    df_trades_benchmark = ms.test_bench_mark('JPM', sd, ed, 100000)
    df_trades_benchmark_transformed = df_trades_transform(df_trades_benchmark)
    portvals_bench = compute_portvals(df_trades_benchmark_transformed, start_val=100000, commission=0, impact=0)
    cum_ret_bench, avg_daily_ret_bench, std_daily_ret_bench, sharpe_ratio_bench = get_portfolio_stats(portvals_bench)

    print(df_trades)
    print(df_trades_transformed)
    print(portvals)
    # Compare portfolio against $SPX
    print(f"Date Range: {sd} to {ed}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of Benchmark : {sharpe_ratio_bench}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of Benchmark : {cum_ret_bench}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of Benchmark : {std_daily_ret_bench}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of Benchmark : {avg_daily_ret_bench}")
    print()
    print(f"Final Portfolio Value of Fund: {portvals[-1]}")
    print(f"Final Portfolio Value of Benchmark: {portvals_bench[-1]}")

    #normalize then plot
    portvals_normalized = portvals/portvals.iloc[0,]
    portvals_bench_normalized = portvals_bench/portvals_bench.iloc[0,]

    plt.plot(portvals_normalized, 'r', label="Theoretically Optimal Portfolio")
    plt.plot(portvals_bench_normalized, 'g', label="Benchmark")
    plt.legend()
    plt.title("Theoretically Optimal Portfolio vs Benchmark (Normalized)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.savefig('Theoretically Optimal Portfolio vs Benchmark (Normalized).png')

if __name__ == "__main__":
    main()
