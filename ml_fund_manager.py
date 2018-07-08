import forecaster as fc
import optimizer as opt
import trader as td
import datetime as dt
import utilities as util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run(n_days=21, max_trade_size=0.10, gen_plot=False):

    delta = 10 #0

    today = dt.date.today()
    yr = today.year
    mo = today.month - 1
    da = today.day - 1

    #end_date = dt.datetime.today() - dt.timedelta(days=delta)
    #start_date = end_date - dt.timedelta(days=1, weeks=52)

    start_date = dt.datetime(yr - 1, mo, da)
    end_date = dt.datetime(yr, mo, da)

    myport, allocations = util.verify_allocations()


    print('-'*20 + '\nFORECAST\n' + '-'*20)
    forecast = fc.forecast(start_date, end_date, myport, n_days=n_days, gen_plot=gen_plot)
    print('\n'+'-'*20 + '\nOPTIMIZE\n' + '-'*20)
    target_allocations = opt.optimize_return(forecast, myport, allocations, gen_plot=gen_plot)
    print('\n' + '-'*20 + '\nORDERS\n' + '-'*20)
    orders = td.create_orders(max_trade_size=max_trade_size)

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
    print("", "Current", "Efficient")
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


if __name__ == "__main__":
    print("Running ML Fund Manager")
    run(n_days=21, max_trade_size=0.2, gen_plot=False)

