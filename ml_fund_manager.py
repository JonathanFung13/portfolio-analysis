import forecaster as fc
import optimizer as opt
import trader as td
import datetime as dt
import utilities as util
import numpy as np
import pandas as pd


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

    forecast = fc.forecast(start_date, end_date, myport, n_days=n_days, gen_plot=gen_plot)
    target_allocations = opt.optimize_return(forecast, myport, allocations, gen_plot=gen_plot)
    orders = td.create_orders(max_trade_size=max_trade_size)

    new_allocations = allocations.copy()
    for i in range(orders.shape[0]):

        index = myport.index(orders.loc[i, 'Symbol'])

        if orders.loc[i, 'Action'] == 'SELL':
            new_allocations[index] -= orders.loc[i, 'Quantity']
        else:
            new_allocations[index] += orders.loc[i, 'Quantity']


#    new_allocations = allocations -


    print(orders)

if __name__ == "__main__":
    print("Running ML Fund Manager")
    run(n_days=21, max_trade_size=0.2, gen_plot=False)

