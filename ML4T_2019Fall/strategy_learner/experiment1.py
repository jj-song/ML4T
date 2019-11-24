import pandas as pd
import numpy as np
from util import get_data, plot_data
import matplotlib.pyplot as plt
import datetime as dt
from strategy_learner.marketsimcode import get_symbols_prices, df_trades_transform, \
    compute_portvals, get_portfolio_stats
from strategy_learner.indicators import get_rolling_mean, get_rolling_std, get_bollinger_bands, get_stochastic
import strategy_learner.ManualStrategy as ms
import strategy_learner.StrategyLearner as sl

def main():
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    symbol = 'JPM'
    sv = 100000

    manual = ms.ManualStrategy()
    strategy = sl.StrategyLearner()

    #df_trades_manual with manual strategy
    df_trades_manual = manual.testPolicy(symbol, sd, ed, 100000)
    df_trades_manual_transformed = df_trades_transform(df_trades_manual, symbol)
    manual_portvals = compute_portvals(df_trades_manual_transformed, start_val=100000, commission=9.95, impact=0.005)
    cum_ret_manual, avg_daily_ret_manual, std_daily_ret_manual, sharpe_ratio_manual = get_portfolio_stats(manual_portvals)

    #df_trades with benchmark (buy once, hold, sell at end)
    df_trades_benchmark = ms.test_bench_mark(symbol, sd, ed, 100000)
    df_trades_benchmark_transformed = df_trades_transform(df_trades_benchmark, symbol)
    portvals_bench = compute_portvals(df_trades_benchmark_transformed, start_val=100000, commission=0, impact=0)
    cum_ret_bench, avg_daily_ret_bench, std_daily_ret_bench, sharpe_ratio_bench = get_portfolio_stats(portvals_bench)

    #df_trades_strategy with strategy learner
    strategy = strategy.StrategyLearner(verbose = False, impact=0.0)
    strategy.addEvidence(symbol, sd, ed, sv)
    df_trades_sl = strategy.testPolicy(symbol, sd, ed, sv)
    sl_portvals = compute_portvals(df_trades_sl, start_val=100000, commission=0, impact=0)
    cum_ret_sl, avg_daily_ret_sl, std_daily_ret_sl, sharpe_ratio_sl = get_portfolio_stats(sl_portvals)



    print(df_trades_manual)
    print(df_trades_manual_transformed)
    print(manual_portvals)
    # Compare portfolio against $SPX
    print(f"Date Range: {sd} to {ed}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio_manual}")
    print(f"Sharpe Ratio of Benchmark : {sharpe_ratio_bench}")
    print(f"Sharpe Ratio of Benchmark : {sharpe_ratio_sl}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret_manual}")
    print(f"Cumulative Return of Benchmark : {cum_ret_bench}")
    print(f"Cumulative Return of Benchmark : {cum_ret_sl}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret_manual}")
    print(f"Standard Deviation of Benchmark : {std_daily_ret_bench}")
    print(f"Standard Deviation of Benchmark : {std_daily_ret_sl}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret_manual}")
    print(f"Average Daily Return of Benchmark : {avg_daily_ret_bench}")
    print(f"Average Daily Return of Benchmark : {avg_daily_ret_sl}")
    print()
    print(f"Final Portfolio Value of Fund: {manual_portvals[-1]}")
    print(f"Final Portfolio Value of Benchmark: {portvals_bench[-1]}")
    print(f"Final Portfolio Value of Benchmark: {sl_portvals[-1]}")

if __name__ == "__main__":
    main()
