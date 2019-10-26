import pandas as pd
import numpy as np
from util import get_data, plot_data
import datetime as dt
from manual_strategy.marketsimcode import get_symbols_prices, autofill_na

class TheoreticallyOptimalStrategy(object):

    def testPolicy(self, symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):

        symbols_prices_df = get_symbols_prices([symbol], sd, ed)
        symbols_prices_df = autofill_na(symbols_prices_df)

        return symbols_prices_df



def main():
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    tos = TheoreticallyOptimalStrategy()
    test = tos.testPolicy('JPM', sd, ed, 100000)
    print(test)


if __name__ == "__main__":
    main()
