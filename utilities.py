import pandas as pd
#import matplotlib.pyplot as plt
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

def extract_stocks(symbols, start, end, addSPY=True): #, colname = 'Close'):
    # Read historical price data from IEX
    # https://pandas-datareader.readthedocs.io/en/latest/remote_data.html
    df = pd.DataFrame(index=pd.date_range(start, end))
    if addSPY and 'SPY' not in symbols: # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        try:
            data = web.DataReader(symbol, 'morningstar', start, end)
            path = os.path.join("data", "{}.csv".format(str(symbol)))
            data.to_csv(path,sep=",",)
        except TypeError:
            print(symbol, "caused an error")
    return data

def extract_vanguard(symbols, start, end):

    duration = str((end - start).days)
    start = start.strftime("%m/%d/%Y")
    end = end.strftime("%m/%d/%Y")

    for symbol in symbols:

        code = vanguard_code(symbol)

        url = "https://advisors.vanguard.com/web/c1/fas-investmentproducts/model.json?paths=[[%27fundReports%27," + \
              code + ",%27price%27,%27fund,etf,vvif%27,[%27to_dt=" + end + ",from_dt=" + start + ",maxDuration=" + \
              duration + "%27,%27to_dt=" + end + ",from_dt=" + start + "%27]]]&method=get"

        json = request_json(url)
        subjson = json['jsonGraph']['fundReports'][code]['price']['value']['price']

        data = pd.DataFrame(data=subjson)
        data = data.rename(columns={'amt': 'Close', 'asOfDt': 'Date'})

        path = os.path.join("data", "{}.csv".format(str(symbol)))
        data.to_csv(path, sep=",",)

        data = data.set_index('Date')

        return data

def load_data(symbols, start, end, addSPY=True, colname = 'Close'):
    df = pd.DataFrame(index=pd.date_range(start, end))
    if addSPY and 'SPY' not in symbols: # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        try:
            path = os.path.join("data", "{}.csv".format(str(symbol)))
            df_temp = pd.read_csv(path, index_col='Date', parse_dates=True,
                                  usecols=['Date', colname], na_values=['nan'])

            # If data files are missing data, redownload the whole range of data.
            if df_temp.index.max() < end: # or df_temp.index.min() > start:
                print("need data")
                if vanguard_code(symbol) != -1:
                    df_temp = extract_vanguard([symbol], start, end)
                else:
                    df_temp = extract_stocks(symbols, start, end)
                df_temp = df_temp[colname].to_frame()

            df_temp = df_temp.rename(columns={colname: symbol})

            df = df.join(df_temp)
            if symbol == 'SPY': # drop dates SPY did not trade
                df = df.dropna(subset=["SPY"])
        except FileNotFoundError:
            print(symbol, "no data found")
    return df


if __name__ == "__main__":

    print("This is just a utility module.  Run portfolio-stats.py")

    if True: # some junk parameters below to help with debugging
        start_date = dt.datetime(2005, 1, 1)
        end_date = dt.datetime(2018, 6, 14)

        target_retirement = [] #['LIKKX']
        stocks = ['DIS', 'TSLA']
        passive_funds = ['VFFSX', 'VITPX', 'VMCPX', 'VSCPX', 'MDDCX','FSPNX', 'VBTIX']
        active_funds = ['FMGEX', 'VMRXX', 'VUSXX']

        symbols = passive_funds + active_funds
        #load_data(stocks, start_date, end_date)
        #load_data(passive_funds, start_date, end_date)
        #extract_vanguard(['VMRXX', 'VUSXX'], start_date, end_date)
        #extract_stocks(stocks, start_date, end_date)
        myport = ['IBM', 'GLD', 'XOM', 'AAPL', 'MSFT', 'TLT', 'SHY']
        extract_stocks(myport, start_date, end_date)


