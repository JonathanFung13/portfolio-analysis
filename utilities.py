import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np
import datetime as dt
#from util import get_data, plot_data
#import scipy.optimize as spo
import pandas_datareader.data as web
import os
#import requests
#from apiclient.discovery import build
#from httplib2 import Http
#from oauth2client import file, client, tools
import requests
#import lxml.html
#from pprint import pprint
#from sys import exit
import json
import numpy as np
#import csv


def vanguard_code(symbol):
    '''
    This is a helper function, there's two functions to get financial data.  One for stocks and the other
    is for Vanguard instruments.  If the symbol sent to this function is from Vanguard a code is returned,
    otherwise a -1 (signalling it is a stock).
    :param symbol: Stock or Mutual Fund symbol
    :return: Vanguard code or -1
    '''
    vanguard_codes = {'VFFSX':'1940', 'VITPX':'871', 'VMCPX':'1859', 'VSCPX':'1861', 'VBTIX':'222',
                      'VMRXX':'66', 'VUSXX':'11'}

    if symbol not in vanguard_codes:
        print("%s is not a valid Vanguard symbol" % symbol)
        return -1
    else:
        return vanguard_codes[symbol]

def request_json(url):
    '''
    This function returns a JSON object from a provided URL.
    '''
    resp_json = requests.get(url).text
    return json.loads(resp_json)

def extract_stocks(symbol, start, end):
    # Read historical price data from IEX
    # https://pandas-datareader.readthedocs.io/en/latest/remote_data.html
    df = pd.DataFrame(index=pd.date_range(start, end))

    try:
        data = web.DataReader(symbol, 'morningstar', start, end)
        data.index = data.index.droplevel(level=0)

        #path = os.path.join("data", "{}.csv".format(str(symbol)))
        #data.to_csv(path,sep=",",)
    except TypeError:
        print(symbol, "caused an error")
    return data

def extract_vanguard(symbol, start, end):

    duration = str((end - start).days)
    start = start.strftime("%m/%d/%Y")
    end = end.strftime("%m/%d/%Y")

    code = vanguard_code(symbol)

    url = "https://advisors.vanguard.com/web/c1/fas-investmentproducts/model.json?paths=[[%27fundReports%27," + \
          code + ",%27price%27,%27fund,etf,vvif%27,[%27to_dt=" + end + ",from_dt=" + start + ",maxDuration=" + \
          duration + "%27,%27to_dt=" + end + ",from_dt=" + start + "%27]]]&method=get"

    json = request_json(url)
    subjson = json['jsonGraph']['fundReports'][code]['price']['value']['price']

    data = pd.DataFrame(data=subjson)
    data = data.rename(columns={'amt': 'Close', 'asOfDt': 'Date'})
    data = data.set_index('Date')
    data = data.sort_index()
    data.index = data.index.to_datetime()

    #path = os.path.join("data", "{}.csv".format(str(symbol)))
    #data.to_csv(path, sep=",",)


    return data

def load_data(symbols, start, end, addSPY=True, colname = 'Close'):
    df = pd.DataFrame(index=pd.date_range(start, end))
    if addSPY and 'SPY' not in symbols: # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        try:
            path = os.path.join("prices", "{}.csv".format(str(symbol)))
            df_temp = pd.read_csv(path, index_col='Date', parse_dates=True, #usecols=['Date', colname],
                                  na_values=['nan', 'null'])
        except FileNotFoundError:
            print(symbol, "no data found")

        # If data files are missing data, redownload the whole range of data.
        if (end - df_temp.index.max()).days > 1: # and end.weekday() < 5: # or df_temp.index.min() > start:
            print("need data")
            if vanguard_code(symbol) != -1:
                df_temp2 = extract_vanguard(symbol, df_temp.index.max()+dt.timedelta(days=1), end)
            else:
                df_temp2 = extract_stocks(symbol, df_temp.index.max()+dt.timedelta(days=1), end)

            df_temp = pd.concat([df_temp, df_temp2], axis=0)
            save_df_as_csv(df_temp, 'prices', symbol, 'Date')

        if 'Adj Close' in df_temp.columns:
            colname = 'Adj Close'
        else:
            colname = 'Close'

        df_temp = df_temp[colname].to_frame()
        df_temp = df_temp.rename(columns={colname: symbol})
        df = df.join(df_temp)
        if symbol == 'SPY': # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])
            start = df.index.min()
            end = df.index.max()
    return df

def calc_daily_er(ers, sf=252):
    return [er/(100*sf) for er in ers]

def compute_returns(prices, allocations, sf=252.0, rfr=0.0):

    expense_ratio = [0.01, 0.03, 0.09, 0.02, 0.03, 0.03, 0.88, 0.05]
    daily_er = calc_daily_er(expense_ratio, 252)


    # Normalize prices and calculate position values
    prices_normalized = prices / prices.ix[0,:]
    prices_normalized = prices_normalized #- daily_er
    position_values = prices_normalized * allocations

    # Get daily portfolio value
    portfolio_value = position_values.sum(axis=1)

    daily_returns = portfolio_value.pct_change()
    daily_returns = daily_returns[1:]


    average_daily_return = daily_returns.mean()
    volatility_daily_return = daily_returns.std()
    sharpe_ratio = (sf**0.5) * (average_daily_return - rfr) / volatility_daily_return

    return average_daily_return, volatility_daily_return, sharpe_ratio, portfolio_value

def save_df_as_csv(df, foldername, filename, indexname=None):

    try:
        path = os.path.join(foldername, "{}.csv".format(str(filename)))
        df.to_csv(path, sep=",", index_label=indexname)
    except TypeError:
        print(filename, "caused an error")

    return

def load_csv_as_df(foldername, filename, index=None):

    path = os.path.join(foldername, "{}.csv".format(filename))

    try:
        if index is None:
            df = pd.read_csv(path)
        elif 'Date' == index:
            df = pd.read_csv(path, index_col=index, parse_dates=True)
        else:
            df = pd.read_csv(path, index_col=index)

        return df
    except:
        return None

def verify_allocations():

    actuals = load_csv_as_df('allocations', 'actual')

    if isinstance(actuals, pd.DataFrame):
        portfolio = actuals.loc[:,'Symbol'].tolist()
        allocations = actuals.loc[:,'Allocations'].tolist()
    else:
        portfolio, allocations = input_allocations()

    return portfolio, allocations


def input_allocations():
    nFunds = int(input('How many funds are in your portfolio? '))

    portfolio = []

    for i in range(nFunds):
        portfolio.append(input(('Enter the symbol for fund number %i: ') % (i+1)))

    print('\n' + '-'*20 + '\n')

    allocations = np.zeros(nFunds)
    for i in range(nFunds):
        allocations[i] = float(input(('Enter the percent allocation for %s: ') % (portfolio[i])))

    if allocations.sum() != 1.0:
        allocations /= allocations.sum()

    allocations = pd.DataFrame(data=allocations, index=portfolio, columns=['Allocations']) #, index=)
    allocations.index.name = 'Symbol'

    save_df_as_csv(allocations, 'allocations', 'actual')

    return portfolio, allocations

def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.grid(linestyle=':')
    plt.show()

if __name__ == "__main__":

    print("This is just a utility module.  Run portfolio-stats.py")

    if True: # some junk parameters below to help with debugging
        start_date = dt.datetime(2005, 1, 1)
        end_date = dt.datetime(2018, 6, 27)

        target_retirement = [] #['LIKKX']
        stocks = ['DIS', 'TSLA']
        passive_funds = ['VFFSX', 'VITPX', 'VMCPX', 'VSCPX', 'MDDCX','FSPNX', 'VBTIX']
        active_funds = ['FMGEX', 'VMRXX', 'VUSXX']

        myfunds = ['VFFSX', 'VITPX', 'VMCPX', 'VSCPX', 'VBTIX']

        #symbols = passive_funds + active_funds
        #load_data(stocks, start_date, end_date)
        #load_data(passive_funds, start_date, end_date)
        #extract_vanguard(myfunds, start_date, end_date)
        extract_stocks(['FSPNX'], start_date, end_date)
        #myport = ['IBM', 'GLD', 'XOM', 'AAPL', 'MSFT', 'TLT', 'SHY']
        #extract_stocks(myport, start_date, end_date)


