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
Input of historical prices, mutual funds available to investor.
Outputs N-day forecast for each mutual fund.
- loads historical data
- transforms historical data for learning
    - new features are technical indicators such as Bollinger Bands, Momentum, etc.

"""

def standard_score(df):
    # Function to standardize a dataframe to mean of 0 and standard deviation of 1
    mean = df.mean()
    sd = df.std()
    return (df - mean) / sd

def melt_indicators(indicator):
    return pd.melt(indicator)

def technical_indicators(start=dt.datetime(2015,1,1), end=dt.datetime(2017,1,1), symbols=['AAPL', 'GOOG'],
                         train_size = 0.7, n_days=21, gen_plot=False, verbose=False):

    prices = util.load_data(symbols, start, end)
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)

    #print('types', prices.dtypes)

    # Create Bollinger Bands & Volatility df
    rolling_mean = prices.rolling(window=n_days+1, center=False).mean()
    rolling_std = prices.rolling(window=n_days+1, center=False).std()
    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std
    bb = (prices - lower_band) / (upper_band - lower_band)

    # Create Momentum df
    momentum = prices.pct_change(5)

    # Create price/sma df
    psma = prices /rolling_mean

    # Create output value
    y = prices.pct_change(n_days)
    y = y.shift(periods=-n_days)

    # Plot all technical indicators
    if gen_plot:
        day1 = 0
        day2 = bb.shape[0]
        for symbol in symbols:
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5)
            #ax1 = prices[symbol].plot(title="Stock prices", fontsize=12)
            line1, = ax1.plot(prices.ix[day1:day2, symbol], color='blue', lw=3, label=symbol)
            line2, = ax1.plot(rolling_mean.ix[day1:day2, symbol], color='grey', lw=1, label='SMA')
            line3, = ax1.plot(upper_band.ix[day1:day2, symbol], color='grey', label='Upper Band')
            line4, = ax1.plot(lower_band.ix[day1:day2, symbol], color='grey', label='Lower Band')
            ax1.grid(linestyle=':')
            ax1.set_title(symbol)

            line5, = ax2.plot(bb.ix[day1:day2, symbol], 'b-')
            ax2.grid(linestyle=':')
            ax2.set_ylabel("Bollinger")

            line6, = ax3.plot(rolling_std.ix[day1:day2, symbol], 'b-')
            ax3.grid(linestyle=':')
            ax3.set_ylabel("Volatil.")

            line7, = ax4.plot(momentum.ix[day1:day2, symbol], 'b-')
            ax4.grid(linestyle=':')
            ax4.set_ylabel("Mom.")

            line8, = ax5.plot(psma.ix[day1:day2, symbol], 'b-')
            ax5.grid(linestyle=':')
            ax5.set_ylabel("$/SMA")

            plt.show()

    if gen_plot:

        fig2, ax = plt.subplots(nrows=bb.shape[1], ncols=1)


        for ax, symbol in zip(ax.flat[0:], bb.columns):
            ax.plot(bb.ix[:, symbol], 'b-')
            ax.set_ylabel(symbol)

        fig2.tight_layout()

        plt.show()

    return bb, rolling_std, momentum, psma, y

def split_data(bb, rolling_std, momentum, psma, y, train_size=0.7, n_days=21, verbose=False):

    # Calculate mean and standard deviation for each indicator in preparation to standardize them
    bb_mean = bb.stack().mean(skipna=True)
    bb_std = bb.stack().std(skipna=True)

    rolling_std_mean = rolling_std.stack().mean(skipna=True)
    rolling_std_std = rolling_std.stack().std(skipna=True)

    momentum_mean = momentum.stack().mean(skipna=True)
    momentum_std = momentum.stack().std(skipna=True)

    psma_mean = psma.stack().mean(skipna=True)
    psma_std = psma.stack().std(skipna=True)

    # Convert all indicators into a single dataframe, one indicator per column
    num_train_rows = round((bb.shape[0] - 2 * n_days) * train_size) + n_days

    train_x1 = melt_indicators(bb[n_days:num_train_rows])
    train_x1 = (train_x1.ix[:, 1] - bb_mean) / bb_std
    train_x2 = melt_indicators(rolling_std[n_days:num_train_rows])
    train_x2 = (train_x2.ix[:, 1] - rolling_std_mean) / rolling_std_std
    train_x3 = melt_indicators(momentum[n_days:num_train_rows])
    train_x3 = (train_x3.ix[:, 1] - momentum_mean) / momentum_std
    train_x4 = melt_indicators(psma[n_days:num_train_rows])
    train_x4 = (train_x4.ix[:, 1] - psma_mean) / psma_std
    train_y = melt_indicators(y[n_days:num_train_rows]).ix[:, 1].to_frame()

    test_x1 = melt_indicators(bb[num_train_rows:-n_days])
    test_x1 = (test_x1.ix[:, 1] - bb_mean) / bb_std
    test_x2 = melt_indicators(rolling_std[num_train_rows:-n_days])
    test_x2 = (test_x2.ix[:, 1] - rolling_std_mean) / rolling_std_std
    test_x3 = melt_indicators(momentum[num_train_rows:-n_days])
    test_x3 = (test_x3.ix[:, 1] - momentum_mean) / momentum_std
    test_x4 = melt_indicators(psma[num_train_rows:-n_days])
    test_x4 = (test_x4.ix[:, 1] - psma_mean) / psma_std
    test_y = melt_indicators(y[num_train_rows:-n_days]).ix[:, 1].to_frame()

    train_x = pd.concat([train_x1, train_x2, train_x3, train_x4],
                         keys=['bollinger', 'volatility', 'momentum', 'psma'], axis=1)

    test_x = pd.concat([test_x1, test_x2, test_x3, test_x4],
                         keys=['bollinger', 'volatility', 'momentum', 'psma'], axis=1)

    forecast_x1 = (bb.ix[-n_days:,1:] - bb_mean) / bb_std
    forecast_x2 = (rolling_std.ix[-n_days:,1:] - rolling_std_mean) / rolling_std_std
    forecast_x3 = (momentum.ix[-n_days:,1:] - momentum_mean) / momentum_std
    forecast_x4 = (psma.ix[-n_days:,1:] - psma_mean) / psma_std

    forecast_x = {}
    for i, symbol in enumerate(forecast_x1.columns):
        forecast_x[symbol] = pd.concat([forecast_x1.ix[:,i], forecast_x2.ix[:,i], forecast_x3.ix[:,i], forecast_x4.ix[:,i]],
                                       keys=['bollinger', 'volatility', 'momentum', 'psma'], axis=1)

    return train_x, test_x, forecast_x, train_y, test_y

def forecast(start=dt.datetime(2015,1,1), end=dt.datetime(2017,1,1), symbols=['AAPL', 'GOOG'],
             train_size = 0.7, n_days=21, max_k=20, gen_plot=False, verbose=False):

    # Load the technical indicators
    bb, rolling_std, momentum, psma, y = technical_indicators(start=start, end=end, symbols=symbols,
                                                           train_size=train_size, n_days=n_days, gen_plot=gen_plot)

    train_x, test_x, forecast_x, train_y, test_y = split_data(bb, rolling_std, momentum, psma, y,
                                                  train_size=train_size, n_days=n_days)



    # construct the set of hyperparameters to tune
    params = {'n_neighbors': np.arange(1, max_k), \
              'weights': ['uniform', 'distance'], \
              'p': [1, 2]}

    classifier = KNeighborsRegressor()  # n_neighbors=n_neighbors, algorithm=algorithm)

    # grid = RandomizedSearchCV(classifier, params)
    grid = GridSearchCV(classifier, params)

    start_timer = time.time()

    grid.fit(train_x, train_y)

    end_timer = time.time() - start_timer

    train_predY = grid.predict(train_x)
    test_predY = grid.predict(test_x)

    if gen_plot:

        fig, axes = plt.subplots(nrows=1, ncols=1)

        axes.set_title('Training/Test Data')
        axes.scatter(train_y, train_predY, c='blue', alpha=0.1)
        axes.scatter(test_y, test_predY, c='green', alpha=0.2)
        axes.set_ylabel("Predicted Daily Return")
        axes.set_xlabel("Actual Daily Return")


        fig.tight_layout()

        plt.show()

        # plt.scatter(train_y, train_predY, c='k')
        # plt.show()
        #
        # plt.scatter(test_y, test_predY, c='k')
        # plt.show()

    train_RMSE = (((train_y - train_predY)**2).sum())**0.5 * 100
    test_RMSE = (((test_y - test_predY)**2).sum())**0.5 * 100


    # evaluate the best grid searched model on the testing data
    if verbose:
        print("[INFO] grid search took {:.2f} seconds".format(end_timer))
        acc = grid.score(test_x, test_y)
        print("[INFO] grid search accuracy: {:.2f}%".format(acc * 100))
        print("[INFO] grid search best parameters: {}".format(
            grid.best_params_))
        print("[Training Error] %.2f" % (train_RMSE))
        print("[Test Error] %.2f" % (test_RMSE))

    days = bb.index[-n_days:] #pd.date_range(end - dt.timedelta(days=n_days-1), end) #bb.index[-n_days:]
    future = pd.date_range(end + dt.timedelta(days=-n_days+1), end)
    forecast_dr = pd.DataFrame(index=days)
    for symbol in symbols:
        forecast_predY = pd.DataFrame(data=grid.predict(forecast_x[symbol]), columns={symbol}, index=days)

#        forecast_temp = forecast_temp.rename(columns={'Return': symbol})
        forecast_dr = forecast_dr.join(forecast_predY)

        #forecast_df = forecast_df.dropna()

    forecast_dr = forecast_dr + 1.0
    first = bb.index[-n_days]

    prices = util.load_data(symbols, first, end)
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    prices = prices[symbols]

    forecast_prices = forecast_dr * prices
    forecast_prices.index.name = 'Date'

    util.save_df_as_csv(forecast_prices, "forecasts", "day_%s_forecast" % (n_days))

    return forecast_dr


if __name__ == "__main__":

    if True:
        start_date = dt.datetime(2017, 6, 27)
        end_date = dt.datetime(2018, 6, 27)
        myport = ['DIS', 'VFFSX', 'VBTIX', 'VITPX', 'VMCPX', 'VSCPX']
        allocations = [0.0, 0.5668 + 0.3879, 0.0453, 0.0, 0.0, 0.0]
        myport = ['VFFSX', 'VBTIX', 'LPSFX', 'VITPX', 'VMCPX', 'VSCPX', 'FMGEX', 'FSPNX']
        allocations = [0.5668, 0.0453, 0.3879, 0.0, 0.0, 0.0, 0.0, 0.0]

        #myfunds = ['DIS']
        #myfunds = ['VFFSX', 'VBTIX']
        #allocations = [0.95, 0.05]

        forecast(start_date, end_date, myport, n_days=21, gen_plot=True)


    else:
        start_date = dt.datetime(2005, 1, 1)
        end_date = dt.datetime(2018, 6, 14)

        myport = ['IBM', 'GLD', 'XOM', 'AAPL', 'MSFT', 'TLT', 'SHY']
        allocations = [0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.02]
        myport = ['IBM', 'GLD']
        allocations = [0.25, 0.75]

        train_x, test_x, train_y, test_y = technical_indicators(start=start_date, end=end_date, symbols=myport,
                                                                train_size = 0.7, n_days=21, gen_plot=False)

