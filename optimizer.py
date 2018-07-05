import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import scipy.optimize as spo
import utilities as util
import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier as knnC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.grid_search import GridSearchCV
import time
import csv
import os
"""
Read forecast, calculate daily prices, then calculate the portfolio stats and use efficient frontier to increase returns

"""

def load_forecasts(start, end, filename="day_21_forecast"):

    #forecast_df = pd.DataFrame(index=pd.date_range(start, end))
    path = os.path.join("forecasts", "{}.csv".format(filename))
    forecast = pd.read_csv(path, index_col='Date', parse_dates=True)
#        forecast_temp = forecast_temp.rename(columns={'Return': symbol})
#        forecast_df = forecast_df.join(forecast_temp)

    return forecast #_df.dropna()

def calc_daily_er(ers, sf=252):
    return [er/(100*sf) for er in ers]

def compute_returns(prices, allocations, sf=252.0, rfr=0.0):

    expense_ratio = [0.01, 0.03, 0.09, 0.02, 0.03, 0.03, 0.88, 0.05]
    daily_er = calc_daily_er(expense_ratio, 252)


    # Normalize prices and calculate position values
    prices_normalized = prices / prices.ix[0,:]
    prices_normalized = prices_normalized - daily_er
    position_values = prices_normalized * allocations

    # Get daily portfolio value
    portfolio_value = position_values.sum(axis=1)

    daily_returns = portfolio_value.pct_change()
    daily_returns = daily_returns[1:]


    average_daily_return = daily_returns.mean()
    volatility_daily_return = daily_returns.std()
    sharpe_ratio = (sf**0.5) * (average_daily_return - rfr) / volatility_daily_return

    return average_daily_return, volatility_daily_return, sharpe_ratio

def optimize_return(forecasts, symbols=['AAPL', 'GOOG'],
                    allocations=[0.5,0.5], rfr=0.0, sf=252.0, gen_plot=False):
    """
    Plot return versus risk for current allocations as well as 500 random allocations.
    Plot return versus risk for each scenario and return the one with the optimal return.
    :param start:
    :param end:
    :param symbols:
    :param allocations:
    :param gen_plot:
    :return:
    """

    # Get statistics for current allocations
    adr_curr, vol_curr, sr_curr = compute_returns(forecasts, allocations=allocations, rfr=rfr, sf=sf)

    # Generate 500 random allocations
    num_allocations = 2000
    adr = [None] * num_allocations
    vol = [None] * num_allocations
    sr = [None] * num_allocations
    iallocations = [None] * num_allocations
    risk_at_max = 0.0
    max_return = 0.0
    sr_max = 0.0

    for i in range(num_allocations):
        weights = np.random.rand(len(symbols))
        iallocations[i] = weights / sum(weights)



    # for i in range(num_allocations - len(symbols)):
    #     weights = np.random.rand(len(symbols))
    #     iallocations[i] = weights / sum(weights)
    #
    # for i in range(len(symbols)):
    #     temp_alloc = [0.0] * len(symbols)
    #     temp_alloc[i] = 1.0
    #     j = num_allocations - len(symbols) + i
    #     iallocations[j] = temp_alloc

    #adr, vol, sr = map(compute_returns(), iallocations)

    for i, allocs in enumerate(iallocations):
        adr[i], vol[i], sr[i] = compute_returns(forecasts, allocations=iallocations[i], rfr=rfr, sf=sf)

        # Logic attempt number 3 for optimizing portfolio: max Sharpe ratio
        if sr[i] > sr_max:
            sr_max = sr[i]
            allocations_ef3 = iallocations[i]

        # Logic attempt number 1 for optimizing portfolio: max return
        if adr[i] > max_return:
            max_return = adr[i]
            risk_at_max = vol[i]
            allocations_ef1 = iallocations[i]
            allocations_ef2 = iallocations[i]


    risk_ef = risk_at_max

    temp_return = adr_curr
    temp_vol = vol_curr

    for i, ireturn in enumerate(adr):
        # Logic attempt number 1 for optimizing portfolio: 90% of max return with lower risk
        if ireturn > (0.9 * max_return) and vol[i] < risk_ef and False:
            risk_ef = vol[i]
            allocations_ef1 = iallocations[i]

        # Logic attempt number 2 for optimizing portfolio: lowest risk with at least same return as current allocation
        if ireturn > adr_curr and vol[i] < temp_vol:
            allocations_ef2 = iallocations[i]
            temp_vol = vol[i]

    allocations_ef4 = np.sum([allocations_ef1, allocations_ef2, allocations_ef3], axis=0)
    allocations_ef4 = np.round(allocations_ef4 / 3, decimals=3)
#    temp = np.array(allocations) * 0.8
#    allocations_ef4 = np.sum([temp, allocations_ef4], axis=0)
    #allocations_ef4 = iallocations[243]

    adr_ef1, vol_ef1, sr_ef1 = compute_returns(forecasts, allocations=allocations_ef1, rfr=rfr, sf=sf)
    adr_ef2, vol_ef2, sr_ef2 = compute_returns(forecasts, allocations=allocations_ef2, rfr=rfr, sf=sf)
    adr_ef3, vol_ef3, sr_ef3 = compute_returns(forecasts, allocations=allocations_ef3, rfr=rfr, sf=sf)
    adr_ef4, vol_ef4, sr_ef4 = compute_returns(forecasts, allocations=allocations_ef4, rfr=rfr, sf=sf)

    print("Portfolios:", "Current", "Efficient")
    print("Daily return: %.5f %.5f %.5f %.5f %.5f" % (adr_curr, adr_ef1, adr_ef2, adr_ef3, adr_ef4))
    print("Daily Risk: %.5f %.5f %.5f %.5f %.5f" % (vol_curr, vol_ef1, vol_ef2, vol_ef3, vol_ef4))
    print("Sharpe Ratio: %.5f %.5f %.5f %.5f %.5f" % (sr_curr, sr_ef1, sr_ef2, sr_ef3, sr_ef4))
    print("Return vs Risk: %.5f %.5f %.5f %.5f %.5f" % (adr_curr/vol_curr, adr_ef1/vol_ef1, adr_ef2/vol_ef2,
                                                        adr_ef3/vol_ef3, adr_ef4/vol_ef4))
    print("\nALLOCATIONS\n" + "-" * 40)
    print("", "Current", "Efficient")
    for i, symbol in enumerate(symbols):
        print("%s %.3f %.3f %.3f %.3f %.3f" %
              (symbol, allocations[i], allocations_ef1[i], allocations_ef2[i], allocations_ef3[i], allocations_ef4[i]))

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:

        fig, ax = plt.subplots()
        ax.scatter(vol, adr, c='blue', s=5, alpha=0.1)
        ax.scatter(vol_curr, adr_curr, c='green', s=15, alpha=0.5) # Current portfolio
        ax.scatter(vol_ef1, adr_ef1, c='red', s=15, alpha=0.5) # ef
        ax.scatter(vol_ef2, adr_ef2, c='orange', s=15, alpha=0.5) # ef
        ax.scatter(vol_ef3, adr_ef3, c='purple', s=15, alpha=0.5) # ef
        ax.scatter(vol_ef4, adr_ef4, c='black', s=25, alpha=0.75) # ef
        ax.set_xlabel('St. Dev. Daily Returns')
        ax.set_ylabel('Mean Daily Returns')
        ax.set_xlim(min(vol)/1.5, max(vol)*1.5)
        ax.set_ylim(min(adr)/1.5, max(adr)*1.5)
        ax.grid()
        ax.grid(linestyle=':')
        fig.tight_layout()
        plt.show()

        # plt.plot(risk, returns, 'o', markersize=5)
        # plt.plot(sddr, adr, 'g+') # Current portfolio
        # plt.plot(sddr_opt, adr_opt, 'b+') # spo optimized
        # plt.plot(risk_at_max, max_return, 'r+') # ef

        # add code to plot here
    #     df_temp = pd.concat([port_val, port_val_opt, port_val_ef, prices_SPY], keys=['Portfolio', 'Optimized', 'EF','SPY'], axis=1)
    #     df_temp = df_temp / df_temp.ix[0, :]
    #     plot_data(df_temp, 'Daily portfolio value and SPY', 'Date', 'Normalized Price')
    #
    # # Add code here to properly compute end value
    # ev = investment * (1+cr)



    target_allocations = pd.DataFrame(data=allocations_ef4, index=symbols, columns=['Allocations']) #, index=)
    target_allocations.index.name = 'Symbol'


    util.save_df_as_csv(target_allocations, 'allocations', 'target')

    return allocations_ef4




if __name__ == "__main__":

    useForecasts = True
    start_date = dt.datetime(2017, 5, 4)
    end_date = dt.datetime(2017, 6, 1)

    myport = ['DIS', 'VFFSX', 'VBTIX', 'VITPX', 'VMCPX', 'VSCPX']
    allocations = [0.0, 0.95, 0.05, 0.0, 0.0, 0.0]
    myport = ['VFFSX', 'VBTIX']
    allocations = [0.95, 0.05]
    myport = ['VFFSX', 'VBTIX', 'LPSFX', 'VITPX', 'VMCPX', 'VSCPX', 'FMGEX', 'FSPNX']
    allocations = [0.5668, 0.0453, 0.3879, 0.0, 0.0, 0.0, 0.0, 0.0]

    if useForecasts:

        forecasts = load_forecasts(start_date, end_date)
        target_allocations = optimize_return(forecasts, myport, allocations, gen_plot=True)

        print(target_allocations)

    else:

        forecasts = load_forecasts(start_date, end_date, "fake_forecast")
        # n_days = 21
        #
        # prices = util.load_data(myfunds, start_date, end_date + dt.timedelta(days=n_days*2))
        # prices.fillna(method='ffill', inplace=True)
        # prices.fillna(method='bfill', inplace=True)
        # prices = prices[myfunds]  # prices of portfolio symbols
        #
        # # Create output value
        # forecasts = prices.pct_change(n_days)
        # forecasts = forecasts.shift(periods=-n_days)
        # forecasts = forecasts[start_date:end_date]

        target_allocations = optimize_return(forecasts, myport, allocations, gen_plot=True)

        print(target_allocations)








