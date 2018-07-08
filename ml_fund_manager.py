import forecaster as fc
import optimizer as opt
import trader as td
import datetime as dt
import utilities as util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calc_start_date(end_date=dt.datetime(2017,1,1), data_size=12):
    return end_date-dt.timedelta(weeks=int(data_size * 52/12))

def run_today(start_date=dt.datetime(2015,1,1), end_date=dt.datetime(2017,1,1), n_days=21, data_size=12,
              myport=['AAPL', 'GOOG'], allocations=[0.5,0.5],
              train_size=0.7, max_k=50, max_trade_size=0.1, gen_plot=False):


    start_date = calc_start_date(end_date, data_size)#end_date - dt.timedelta(weeks=int(data_size * 52/12))

    #myport, allocations = util.verify_allocations()


    print('-'*20 + '\nFORECAST\n' + '-'*20)
    forecast = fc.forecast(start_date, end_date, symbols=myport, train_size=train_size,
                           n_days=n_days, max_k=max_k, gen_plot=gen_plot)

    print('\n'+'-'*20 + '\nOPTIMIZE\n' + '-'*20)
    target_allocations = opt.optimize_return(forecast, myport, allocations, gen_plot=gen_plot)

    print('\n' + '-'*20 + '\nORDERS\n' + '-'*20)
    orders = td.create_orders(allocations, target_allocations, max_trade_size=max_trade_size)

    print(orders)

    new_allocations = allocations.copy()
    for i in range(orders.shape[0]):

        index = myport.index(orders.loc[i, 'Symbol'])

        if orders.loc[i, 'Action'] == 'SELL':
            new_allocations[index] -= orders.loc[i, 'Quantity']
        else:
            new_allocations[index] += orders.loc[i, 'Quantity']


    adr_current, vol_current, sr_current, pv_current = util.compute_returns(forecast, allocations=allocations)
    adr_target, vol_target, sr_target, pv_target = util.compute_returns(forecast, allocations=target_allocations)
    adr_new, vol_new, sr_new, pv_new = util.compute_returns(forecast, allocations=new_allocations)

    print("Portfolios:", "Current", "Target","New")
    print("Daily return: %.5f %.5f %.5f" % (adr_current, adr_target, adr_new))
    print("Daily Risk: %.5f %.5f %.5f" % (vol_current, vol_target, vol_new))
    print("Sharpe Ratio: %.5f %.5f %.5f" % (sr_current, sr_target, sr_new))
    print("Return vs Risk: %.5f %.5f %.5f" % (adr_current/vol_current, adr_target/vol_target, adr_new/vol_new))
    print("\nALLOCATIONS\n" + "-" * 40)
    print("Symbol", "Current", "Target", 'New')
    for i, symbol in enumerate(myport):
        print("%s %.3f %.3f %.3f" %
              (symbol, allocations[i], target_allocations[i], new_allocations[i]))

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot or True:

        fig, ax = plt.subplots()
        ax.scatter(vol_current, adr_current, c='green', s=15, alpha=0.5) # Current portfolio
        ax.scatter(vol_target, adr_target, c='red', s=15, alpha=0.5) # ef
        ax.scatter(vol_new, adr_new, c='black', s=25, alpha=0.75) # ef
        ax.set_xlabel('St. Dev. Daily Returns')
        ax.set_ylabel('Mean Daily Returns')
        #ax.set_xlim(min(vol)/1.5, max(vol)*1.5)
        #ax.set_ylim(min(adr)/1.5, max(adr)*1.5)
        ax.grid()
        ax.grid(linestyle=':')
        fig.tight_layout()
        plt.show()


        # add code to plot here
        df_temp = pd.concat([pv_current, pv_target, pv_new], keys=['Current', 'Target', 'New'], axis=1)
        df_temp = df_temp / df_temp.ix[0, :]
        util.plot_data(df_temp, 'Forecasted Daily portfolio value and SPY', 'Date-21', 'Normalized Price')


    prior_prices = util.load_data(myport, start_date, end_date)
    prior_prices.fillna(method='ffill', inplace=True)
    prior_prices.fillna(method='bfill', inplace=True)


    #prices_SPY = prior_prices['SPY']  # SPY prices, for benchmark comparison
    prior_prices = prior_prices[myport]  # prices of portfolio symbols

    forecast_prices = forecast * prior_prices

    time_span = pd.date_range(forecast.index.min(), end_date + dt.timedelta(days=n_days*2))
    forecast_prices = forecast_prices.reindex(time_span)
    forecast_prices = forecast_prices.shift(periods=n_days*2)
    forecast_prices = forecast_prices.dropna()

    forecast_prices = pd.concat([prior_prices, forecast_prices], axis=0)

    adr_current, vol_current, sr_current, pv_current = util.compute_returns(forecast_prices, allocations=allocations)
    adr_target, vol_target, sr_target, pv_target = util.compute_returns(forecast_prices, allocations=target_allocations)
    adr_new, vol_new, sr_new, pv_new = util.compute_returns(forecast_prices, allocations=new_allocations)

    df_temp = pd.concat([pv_current, pv_target, pv_new], keys=['Current', 'Target', 'New'], axis=1)
    df_temp = df_temp / df_temp.ix[0, :]
    util.plot_data(df_temp, 'Daily portfolio value and SPY', 'Date', 'Normalized Price')

    return adr_new, vol_new, sr_new, pv_new, new_allocations

def test_experiment_one(n_days=21, data_size=12, train_size=0.7, max_k=50, max_trade_size=0.1,
                        years_to_go_back=2, gen_plot=False):

    today = dt.date.today()
    yr = today.year - years_to_go_back
    mo = today.month - 1  # Just temporary, take out 1 when data download is fixed.
    da = today.day - 1

    end_date = dt.datetime(yr, mo, da)

    adr = [None] * 12
    vol = [None] * 12
    sr = [None] * 12

    myport = ['AAPL', 'GLD']
    myalloc = [0.5,0.5]

    for i in range(12):
        end_date = end_date + dt.timedelta(weeks=int((i+1)*52/12))

        print(('EXPERIMENT %i') % (i))

        adr[i], vol[i], sr[i], pv, myalloc = run_today(end_date=end_date, n_days=n_days, data_size=data_size,
                                                       myport=myport, allocations=myalloc,
                                                       train_size=train_size, max_k=max_k,
                                                       max_trade_size=max_trade_size, gen_plot=False)


if __name__ == "__main__":

    test = True

    today = dt.date.today()
    yr = today.year
    mo = today.month - 1  # Just temporary, take out 1 when data download is fixed.
    da = today.day - 1

    end_date = dt.datetime(yr, mo, da)

    if test == False:
        print("Running ML Fund Manager")

        myport, allocations = util.verify_allocations()

        n_days = 21 # How long the forecast should look out
        data_size = 12 # Number of months of data to use for forecasting
        train_size = 0.70 # Size of training data, rest is test
        max_k = 50 # Maximum value of k for kNN
        max_trade_size= 0.10 # Maximum amount of allocation allowed in a trade

        run_today(end_date=end_date, n_days=n_days, data_size=data_size,
                  myport=myport, allocations=allocations,
                  train_size=train_size, max_k=max_k,
                  max_trade_size=max_trade_size, gen_plot=False)
    else:
        print("Testing ML Fund Manager")

        n_days = 21  # How long the forecast should look out
        data_size = 12  # Number of months of data to use for forecasting
        train_size = 0.70  # Size of training data, rest is test
        max_k = 50  # Maximum value of k for kNN
        max_trade_size = 0.10  # Maximum amount of allocation allowed in a trade

        years_to_go_back = 2

        test_experiment_one(n_days=n_days, data_size=data_size, train_size=train_size, max_k=max_k,
                            max_trade_size=max_trade_size, years_to_go_back=years_to_go_back, gen_plot=False)

