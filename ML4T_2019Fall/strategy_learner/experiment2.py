import pandas as pd
import numpy as np
from util import get_data, plot_data
import matplotlib.pyplot as plt
import datetime as dt
from marketsimcode import get_symbols_prices, df_trades_transform, \
    compute_portvals, get_portfolio_stats, normalize
import ManualStrategy as ms
import StrategyLearner as sl

def main():
    sd_train = dt.datetime(2008,1,1)
    ed_train = dt.datetime(2009,12,31)
    sd = dt.datetime(2010,1,1) #uncomment this for insample test
    ed = dt.datetime(2011,12,31) #uncomment this for insample test

    symbol = 'JPM'
    sv = 100000

    manual_learner = ms.ManualStrategy()
    strategy_learner = sl.StrategyLearner(verbose=False, impact=0.05)

    #df_trades_strategy with strategy learner
    strategy_learner.addEvidence(symbol = symbol, sd=sd_train, ed=ed_train, sv = 100000) # training phase #by default will always run from 2008 to 2009 12/31
    df_trades_sl = strategy_learner.testPolicy(symbol, sd, ed, sv) #change sd & ed to test out sample
    df_trades_sl['Symbol'] = symbol
    df_trades_sl.loc[df_trades_sl.Shares > 0, 'Order'] = 'BUY'
    df_trades_sl.loc[df_trades_sl.Shares < 0, 'Order'] = 'SELL'
    df_trades_sl = df_trades_sl[df_trades_sl.Shares !=0]


    sl_portvals = compute_portvals(df_trades_sl, start_val=sv, commission=0, impact=0.05)
    cum_ret_sl, avg_daily_ret_sl, std_daily_ret_sl, sharpe_ratio_sl = get_portfolio_stats(sl_portvals)

    #df_trades_manual with manual strategy
    df_trades_manual = manual_learner.testPolicy(symbol, sd, ed, sv)
    df_trades_manual_transformed = df_trades_transform(df_trades_manual, symbol)
    manual_portvals = compute_portvals(df_trades_manual_transformed, start_val=sv, commission=0, impact=0.05)
    cum_ret_manual, avg_daily_ret_manual, std_daily_ret_manual, sharpe_ratio_manual = get_portfolio_stats(manual_portvals)

    #df_trades with benchmark (buy once, hold, sell at end)
    df_trades_benchmark = manual_learner.test_bench_mark(symbol, sd, ed, sv)
    df_trades_benchmark_transformed = df_trades_transform(df_trades_benchmark, symbol)
    portvals_bench = compute_portvals(df_trades_benchmark_transformed, start_val=sv, commission=0, impact=0.05)
    cum_ret_bench, avg_daily_ret_bench, std_daily_ret_bench, sharpe_ratio_bench = get_portfolio_stats(portvals_bench)

    joined = normalize(manual_portvals).to_frame().join(normalize(sl_portvals).to_frame(), lsuffix="m", rsuffix="s").\
        join(normalize(portvals_bench).to_frame(), lsuffix="ms", rsuffix="b")

    joined.columns = ["Manual Strategy Portfolio", "Strategy Learner", "Benchmark"]
    fig = joined.plot(title="Manual Strategy vs Strategy Learner vs Benchmark with Impact",
                      fontsize=12, lw=1, color=["red", "green", "blue"])
    fig.set_xlabel("Date")
    fig.set_ylabel("Price")
    plt.tight_layout()
    plt.savefig("Manual Strategy vs Strategy Learner vs Benchmark with Impact")
    plt.clf()


    print(df_trades_manual)
    print(df_trades_manual_transformed)
    print(manual_portvals)
    # Compare portfolio against $SPX
    print(f"Date Range: {sd} to {ed}")
    print()
    print(f"Sharpe Ratio of Manual Strategy: {sharpe_ratio_manual}")
    print(f"Sharpe Ratio of Benchmark : {sharpe_ratio_bench}")
    print(f"Sharpe Ratio of Strategy Learner : {sharpe_ratio_sl}")
    print()
    print(f"Cumulative Return of Manual Strategy: {cum_ret_manual}")
    print(f"Cumulative Return of Benchmark : {cum_ret_bench}")
    print(f"Cumulative Return of Strategy Learner : {cum_ret_sl}")
    print()
    print(f"Standard Deviation of Manual Strategy: {std_daily_ret_manual}")
    print(f"Standard Deviation of Benchmark : {std_daily_ret_bench}")
    print(f"Standard Deviation of Strategy Learner : {std_daily_ret_sl}")
    print()
    print(f"Average Daily Return of Manual Strategy: {avg_daily_ret_manual}")
    print(f"Average Daily Return of Benchmark : {avg_daily_ret_bench}")
    print(f"Average Daily Return of Strategy Learner : {avg_daily_ret_sl}")
    print()
    print(f"Final Portfolio Value of Manual Strategy: {manual_portvals[-1]}")
    print(f"Final Portfolio Value of Benchmark: {portvals_bench[-1]}")
    print(f"Final Portfolio Value of Strategy Learner: {sl_portvals[-1]}")

def author():
    return 'jsong350'

if __name__ == "__main__":
    main()
