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

def optimize_return(forecasts, symbols=['AAPL', 'GOOG'],
                    allocations=[0.5,0.5], rfr=0.0, sf=252.0, gen_plot=False, verbose=False, savelogs=False):
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
    adr_curr, vol_curr, sr_curr, pv_curr = util.compute_returns(forecasts, allocations=allocations, rfr=rfr, sf=sf)

    # Generate n random allocations
    num_allocations = 2000
    iallocations = [None] * num_allocations
    for i in range(num_allocations):
        weights = np.random.rand(len(symbols))
        iallocations[i] = weights / sum(weights)

    # Generate allocations for 100% in each of the available funds
    for i in range(len(symbols)):
        temp_alloc = [0.0] * len(symbols)
        temp_alloc[i] = 1.0
        iallocations.append(temp_alloc)

    num_allocations += len(symbols)
    adr = [None] * num_allocations
    vol = [None] * num_allocations
    sr = [None] * num_allocations

    risk_at_max = 100.0
    max_return = -100.0
    sr_max = -100.0

    #adr, vol, sr = map(compute_returns(), iallocations)

    for i, allocs in enumerate(iallocations):
        adr[i], vol[i], sr[i], pv_i = util.compute_returns(forecasts, allocations=iallocations[i], rfr=rfr, sf=sf)

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

    if verbose or gen_plot:
        adr_ef1, vol_ef1, sr_ef1, pv_ef1 = util.compute_returns(forecasts, allocations=allocations_ef1, rfr=rfr, sf=sf)
        adr_ef2, vol_ef2, sr_ef2, pv_ef2 = util.compute_returns(forecasts, allocations=allocations_ef2, rfr=rfr, sf=sf)
        adr_ef3, vol_ef3, sr_ef3, pv_ef3 = util.compute_returns(forecasts, allocations=allocations_ef3, rfr=rfr, sf=sf)
        adr_ef4, vol_ef4, sr_ef4, pv_ef4 = util.compute_returns(forecasts, allocations=allocations_ef4, rfr=rfr, sf=sf)

    if verbose and False: # not going to print these out from here anymore
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
        ax.scatter(vol_curr, adr_curr, c='green', s=35, alpha=0.75) # Current portfolio
        ax.scatter(vol_ef1, adr_ef1, c='red', s=35, alpha=0.5) # ef
        ax.scatter(vol_ef2, adr_ef2, c='red', s=35, alpha=0.5) # ef
        ax.scatter(vol_ef3, adr_ef3, c='red', s=35, alpha=0.5) # ef
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

    if savelogs:
        target_allocations = pd.DataFrame(data=allocations_ef4, index=symbols, columns=['Allocations']) #, index=)
        target_allocations.index.name = 'Symbol'

        util.save_df_as_csv(target_allocations, 'logs', 'target', 'Symbol')

    return allocations_ef4




if __name__ == "__main__":
    print("Run ml_fund_manager.py instead")
