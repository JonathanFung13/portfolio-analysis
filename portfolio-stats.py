
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

def download_stocks(symbols, start, end, addSPY=True): #, colname = 'Close'):
    # Read historical price data from IEX
    # https://pandas-datareader.readthedocs.io/en/latest/remote_data.html
    df = pd.DataFrame(index=pd.date_range(start, end))
    if addSPY and 'SPY' not in symbols: # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        try:
            f = web.DataReader(symbol, 'morningstar', start, end)
            path = os.path.join("data", "{}.csv".format(str(symbol)))
            f.to_csv(path,sep=",",)
        except TypeError:
            print(symbol, "caused an error")
    return

def request_json(url):
    resp_json = requests.get(url).text
    return json.loads(resp_json)

def download_vg(symbols):#, start, end):
    temp = 'https://advisors.vanguard.com/web/c1/fas-investmentproducts/model.json?paths=%5B%5B%27literatureDetail%27%2C%271940%27%5D%2C%5B%27priceServiceData%27%2C%27VFFSX%27%5D%2C%5B%27fundReports%27%2C1940%2C%5B%27distribution%27%2C%27statistics%27%5D%2C%27fund%2Cetf%2Cvvif%27%5D%2C%5B%27fundReports%27%2C1940%2C%27distribution%27%2C%27fund%2Cetf%2Cvvif%27%2C%27from_dt%3D06%2F23%2F2016%2Cto_dt%3D05%2F29%2F2018%27%5D%2C%5B%27fundReports%27%2C1940%2C%27price%27%2C%27fund%2Cetf%2Cvvif%27%2C%5B%27to_dt%3D05%2F29%2F2018%2Cfrom_dt%3D05%2F27%2F2016%2CmaxDuration%3D732%27%2C%27to_dt%3D05%2F29%2F2018%2Cfrom_dt%3D05%2F29%2F2017%27%5D%5D%2C%5B%27search%27%2C%27related-articles%27%2C%27VFFSX%27%2C%27mutual%20fund%27%2C%27equity%27%5D%5D&method=get'
    temp = 'https://advisors.vanguard.com/web/c1/fas-investmentproducts/model.json?paths=%5B%5B%27fundReports%27%2C1940%2C%5B%27price%27%2C%27statistics%27%5D%2C%27fund%2Cetf%2Cvvif%27%5D%2C%5B%27fundReports%27%2C1940%2C%27price%27%2C%27fund%2Cetf%2Cvvif%27%2C%27from_dt%3D03%2F23%2F2018%2Cto_dt%3D05%2F29%2F2018%27%5D%5D&method=get'
    json = request_json(temp)
    subjson = json['jsonGraph']['fundReports']['1940']['price']['value']['price']
    price = pd.DataFrame(data=subjson)
    print('hello')

#temp = 'https://advisors.vanguard.com/web/c1/fas-investmentproducts/model.json?paths=%5B%5B%27literatureDetail%27%2C%271940%27%5D%2C%5B%27priceServiceData%27%2C%27VFFSX%27%5D%2C%5B%27fundReports%27%2C1940%2C%5B%27distribution%27%2C%27statistics%27%5D%2C%27fund%2Cetf%2Cvvif%27%5D%2C%5B%27fundReports%27%2C1940%2C%27distribution%27%2C%27fund%2Cetf%2Cvvif%27%2C%27from_dt%3D06%2F23%2F2016%2Cto_dt%3D05%2F29%2F2018%27%5D%2C%5B%27fundReports%27%2C1940%2C%27price%27%2C%27fund%2Cetf%2Cvvif%27%2C%5B%27to_dt%3D05%2F29%2F2018%2Cfrom_dt%3D05%2F27%2F2016%2CmaxDuration%3D732%27%2C%27to_dt%3D05%2F29%2F2018%2Cfrom_dt%3D05%2F29%2F2017%27%5D%5D%2C%5B%27search%27%2C%27related-articles%27%2C%27VFFSX%27%2C%27mutual%20fund%27%2C%27equity%27%5D%5D&method=get'


def read_data(symbols, start, end, addSPY=True, colname = 'Close'):
    df = pd.DataFrame(index=pd.date_range(start, end))
    if addSPY and 'SPY' not in symbols: # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        try:
            path = os.path.join("data", "{}.csv".format(str(symbol)))
            df_temp = pd.read_csv(path, index_col='Date', parse_dates=True,
                                  usecols=['Date', colname], na_values=['nan'])
            df_temp = df_temp.rename(columns={colname: symbol})
            df = df.join(df_temp)
            if symbol == 'SPY': # drop dates SPY did not trade
                df = df.dropna(subset=["SPY"])
        except FileNotFoundError:
            print(symbol, "no data found")
    return df

def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    #    ax.grid(linestyle=':')
    plt.show()

def compute_portfolio_value(stard, end, symbols, allocations, rfr=0.0, sf =252.0)
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices_all.fillna(method='ffill', inplace=True)
    prices_all.fillna(method='bfill', inplace=True)
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Normalize the prices according to the first day. The first row
    # for each stock should have a value of 1.0 at this point.
    normed = prices / prices.ix[0,:]

    # Multiply each column by the allocation to the corresponding equity.
    alloced = normed * allocs

    # Multiply these normalized allocations by starting value of overall portfolio, to get position values.
    pos_vals = alloced * sv

    # Get daily portfolio value
    port_val = pos_vals.sum(axis=1)

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = compute_portfolio_stats(port_val, rfr=rfr, sf=sf)

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
    normed = prices / prices.ix[0,:]
    alloced = normed * allocs
    pos_vals = alloced * 1.0 #use 1 as starting value

    # Get daily portfolio value
    port_val = pd.DataFrame({'Total': pos_vals.sum(axis=1)}) # add code here to compute daily portfolio values

#    cr = (port_val[-1] / port_val[0]) - 1
    cr = port_val['Total'].iloc[-1] - 1

    daily_returns = port_val.pct_change()
    daily_returns = daily_returns[1:]

    adr = daily_returns['Total'].mean()
    sddr = daily_returns['Total'].std()

    sr = (sf**0.5) * (adr - rfr) / sddr

    return cr, adr, sddr, sr

if __name__ == "__main__":

    start_date = dt.datetime(2018, 4, 1)
    end_date = dt.datetime(2018, 5, 1)

    target_retirement = [] #['LIKKX']
    passive_funds = ['DIS', 'TSLA','VFFSX', 'VITPX', 'VMCPX', 'VSCPX', 'MDDCX','FSPNX', 'VBTIX']
    active_funds = ['FMGEX', 'VMRXX', 'VUSXX']

    symbols = passive_funds + active_funds

    #download_stocks(symbols, start_date, end_date)
    download_vg('1940')
    prices = read_data(symbols, start_date, end_date)

    # Assess the portfolio
    cr, adr, sddr, sr, ev = compute_portfolio_stats(sd=start_date, ed=end_date, \
                                                        syms=symbols, \
                                                        gen_plot=True)
