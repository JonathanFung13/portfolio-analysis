
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
#from util import get_data, plot_data
import scipy.optimize as spo
import pandas_datareader.data as web
import os
import requests
from apiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
import requests
#import lxml.html
from pprint import pprint
#from sys import exit
import json
#import csv
import data_utilities as du



def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    #    ax.grid(linestyle=':')
    plt.show()

def compute_portfolio_value(start, end, symbols, allocations, rfr=0.0, sf =252.0, gen_plot=False):
    # Read in adjusted closing prices for given symbols, date range
    sv = 1000
    dates = pd.date_range(start, end)
    prices_all = du.load_data(symbols, start, end)  # automatically adds SPY
    prices_all.fillna(method='ffill', inplace=True)
    prices_all.fillna(method='bfill', inplace=True)
    prices = prices_all[symbols]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Normalize the prices according to the first day. The first row
    # for each stock should have a value of 1.0 at this point.
    normed = prices / prices.ix[0,:]

    # Multiply each column by the allocation to the corresponding equity.
    alloced = normed * allocations

    # Multiply these normalized allocations by starting value of overall portfolio, to get position values.
    pos_vals = alloced * sv

    # Get daily portfolio value
    port_val = pos_vals.sum(axis=1)

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = compute_returns(port_val, rfr=rfr, sf=sf)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_temp = df_temp / df_temp.ix[0,:]
        plot_data(df_temp, 'Daily portfolio value and SPY', 'Date', 'Normalized Price')

    # Add code here to properly compute end value
    ev = sv * (1+cr)

    return cr, adr, sddr, sr, ev

def compute_returns(prices, \
    allocs=[0.1,0.2,0.3,0.4], \
    rfr=0.0, sf=252.0):

    # Normalize prices and calculate position values
    # normed = prices / prices.ix[0,:]
    # alloced = normed * allocs
    # pos_vals = alloced * 1.0 #use 1 as starting value
    #
    # # Get daily portfolio value
    # port_val = pd.DataFrame({'Total': pos_vals.sum(axis=1)}) # add code here to compute daily portfolio values

    port_val = prices
    cr = (port_val[-1] / port_val[0]) - 1
#    cr = port_val['Total'].iloc[-1] - 1

    daily_returns = port_val.pct_change()
    daily_returns = daily_returns[1:]

    # adr = daily_returns['Total'].mean()
    # sddr = daily_returns['Total'].std()
    adr = daily_returns.mean()
    sddr = daily_returns.std()

    sr = (sf**0.5) * (adr - rfr) / sddr

    return cr, adr, sddr, sr

if __name__ == "__main__":

    start_date = dt.datetime(2016, 1, 1)
    end_date = dt.datetime(2018, 6, 1)

    target_retirement = [] #['LIKKX']
    passive_funds = ['DIS', 'TSLA','VFFSX', 'VITPX', 'VMCPX', 'VSCPX', 'MDDCX','FSPNX', 'VBTIX']
    active_funds = ['FMGEX', 'VMRXX', 'VUSXX']
    symbols = passive_funds + active_funds

    myport = ['DIS', 'TSLA','VFFSX', 'VBTIX', 'BTCLP40']
    allocations = [0.0, 0.0, 0.5668, 0.0453, 0.3879]

#    prices = read_data(symbols, start_date, end_date)

    # Assess the portfolio
    cr, adr, sddr, sr, ev = compute_portfolio_value(start_date, end_date, myport, allocations, \
                                                        gen_plot=True)

    print(cr, adr, sddr, sr, ev)
