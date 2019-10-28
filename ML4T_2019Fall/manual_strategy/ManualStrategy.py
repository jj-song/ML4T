import pandas as pd
import numpy as np
from util import get_data, plot_data
import matplotlib.pyplot as plt
import datetime as dt
from manual_strategy.marketsimcode import get_symbols_prices, df_trades_transform, \
    compute_portvals, get_portfolio_stats
from manual_strategy.indicators import get_rolling_mean, get_rolling_std, get_bollinger_bands

class ManualStrategy (object):

    def __init__(self):
        self.short_entry = []
        self.long_entry = []

    def test_bench_mark(self, symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):
        symbols_prices_df = get_symbols_prices([symbol], sd, ed)
        adj_sd = symbols_prices_df.index[0] #don't remove. date may change depending on if market open
        adj_ed = symbols_prices_df.index[-1]
        df_trades_benchmark = pd.DataFrame(data= [1000, -1000], index = [adj_sd, adj_ed], columns = ['orders'])

        return df_trades_benchmark

    def testPolicy(self, symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):
        symbols_prices_df = get_symbols_prices([symbol], sd, ed)
        window = 20
        net_holdings = 0
        order_size = []
        order_date = []
        prices = symbols_prices_df[symbol]

        #bollinger band strategy
        rolling_mean = get_rolling_mean(prices, window)
        rolling_std = get_rolling_std(prices, window)
        upper_band, lower_band = get_bollinger_bands(rolling_mean, rolling_std)

        #get new dataframe using bollinger band and adjust for nan days
        upper_band_without_nan = upper_band[upper_band.notnull()].to_frame()
        upper_band_without_nan_series = upper_band[upper_band.notnull()]
        lower_band_without_nan = lower_band[lower_band.notnull()].to_frame()
        lower_band_without_nan_series = lower_band[lower_band.notnull()]
        sd_adj_for_nan = upper_band_without_nan.index[0]
        ed_adj_for_nan = upper_band_without_nan.index[-1]

        symbols_prices_df_adj_for_nan = get_symbols_prices([symbol], sd_adj_for_nan, ed_adj_for_nan)
        prices_adj_for_nan = symbols_prices_df_adj_for_nan[symbol]
        total_number_of_trading_days = len(symbols_prices_df_adj_for_nan.index)

        for day in range(total_number_of_trading_days-1): # minus 1 to offset last day
            day = day+1 #ignore first day since -1 will get last day of series
            day_before = day-1
            date_of_order = symbols_prices_df_adj_for_nan.index[day]

            upper_band_day_before_adj_for_nan = upper_band_without_nan_series[day_before]
            upper_band_day_adj_for_nan = upper_band_without_nan_series[day]


            lower_band_day_before_adj_for_nan = lower_band_without_nan_series[day_before]
            lower_band_day_adj_for_nan = lower_band_without_nan_series[day]

            price_day_before_adj_for_nan = prices_adj_for_nan[day_before]
            price_adj_for_nan = prices_adj_for_nan[day]

            #check if price of stock day before is higher than the upper bb day before
            #check if price of stock day is lower than the upper bb
            if price_day_before_adj_for_nan > upper_band_day_before_adj_for_nan \
                    and price_adj_for_nan < upper_band_day_adj_for_nan \
                    and net_holdings != -1000:
                order_date.append(date_of_order)
                order_size.append(-1000-net_holdings)
                net_holdings = -1000
                self.short_entry.append(date_of_order)
            elif price_day_before_adj_for_nan < lower_band_day_before_adj_for_nan \
                    and price_adj_for_nan > lower_band_day_adj_for_nan \
                    and net_holdings != 1000:
                order_date.append(date_of_order)
                order_size.append(1000-net_holdings)
                net_holdings = 1000
                self.long_entry.append(date_of_order)



        if net_holdings in (1000, -1000): # sell any outstanding share to calculate total cash value of port on last day
            order_date.append(symbols_prices_df.index[-1])
            order_size.append(- net_holdings)

        df_trades = pd.DataFrame(data = order_size, index = order_date, columns = ['orders'])

        return df_trades


def main():
    sd = dt.datetime(2010,1,1)
    ed = dt.datetime(2011,12,31)
    ms = ManualStrategy()
    symbol = 'JPM'

    #df_trades with optimal strategy
    df_trades = ms.testPolicy(symbol, sd, ed, 100000)
    df_trades_transformed = df_trades_transform(df_trades, symbol)
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
    plt.savefig('Manual Strategy vs Benchmark (Normalized).png')

if __name__ == "__main__":
    main()
