
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import scipy.optimize as spo
import data_utilities as du



def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.grid(linestyle=':')
    plt.show()

def compute_portfolio_value(start, end, symbols, allocations, rfr=0.0, sf =252.0, gen_plot=False):
    # Read in adjusted closing prices for given symbols, date range
    investment = 1000 # in dollars

    prices = du.load_data(symbols, start, end)
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    prices_SPY = prices['SPY']  # SPY prices, for benchmark comparison
    prices = prices[symbols]  # prices of portfolio symbols

    # Get portfolio statistics
    cr, adr, sddr, sr, port_val = compute_returns(prices, allocations=allocations, rfr=rfr, sf=sf)

    optimized_allocations = optimal_allocations(prices)
    cr_opt, adr_opt, sddr_opt, sr_opt, port_val_opt = compute_returns(prices, allocations=optimized_allocations, rfr=rfr, sf=sf)

    print("ALLOCATIONS\n" + "-" * 40)
    for i, symbol in enumerate(symbols):
        print("%s actual: %.2f optimal: %.2f" % (symbol, allocations[i], optimized_allocations[i]))


    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, port_val_opt,prices_SPY], keys=['Portfolio', 'Optimized','SPY'], axis=1)
        df_temp = df_temp / df_temp.ix[0,:]
        plot_data(df_temp, 'Daily portfolio value and SPY', 'Date', 'Normalized Price')

    # Add code here to properly compute end value
    ev = investment * (1+cr)

    return cr, adr, sddr, sr, ev

def compute_returns(prices, \
    allocations=[0.1,0.2,0.3,0.4], \
    investment = 1.0,
    rfr=0.0, sf=252.0):

    # Normalize prices and calculate position values
    prices_normalized = prices / prices.ix[0,:]
    allocated = prices_normalized * allocations
    position_values = allocated * investment

    # Get daily portfolio value
    portfolio_value = position_values.sum(axis=1)

    cr = (portfolio_value[-1] / portfolio_value[0]) - 1

    daily_returns = portfolio_value.pct_change()
    daily_returns = daily_returns[1:]

    adr = daily_returns.mean()
    sddr = daily_returns.std()

    sharpe_ratio = (sf**0.5) * (adr - rfr) / sddr

    return cr, adr, sddr, sharpe_ratio, portfolio_value

def optimal_allocations(prices):
    guess = [1.0 / prices.shape[1]] * prices.shape[1]
    bounds = tuple((0.0, 1.0) for x in range(prices.shape[1]))
    constrain = ({'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)})
    return spo.minimize(compute_sharpe_ratio, guess, args=(prices,), method='SLSQP', bounds=bounds, constraints=constrain).x

def compute_sharpe_ratio(Allocations, prices):
    return -compute_returns(prices, Allocations)[3]

if __name__ == "__main__":

    month = 1
    year = 2018
    start_date = dt.datetime(year, 1, 1)
    end_date = dt.datetime(year, 6, 1)

    target_retirement = [] #['LIKKX']
    passive_funds = ['DIS', 'TSLA','VFFSX', 'VITPX', 'VMCPX', 'VSCPX', 'MDDCX','FSPNX', 'VBTIX']
    active_funds = ['FMGEX', 'VMRXX', 'VUSXX']
    symbols = passive_funds + active_funds

    myport = ['VFFSX', 'VBTIX', 'BTCLP40', 'VITPX', 'VMCPX', 'VSCPX']
    allocations = [0.5668, 0.0453, 0.3879, 0.0, 0.0, 0.0]

#    prices = read_data(symbols, start_date, end_date)

    # Assess the portfolio
    cr, adr, sddr, sr, ev = compute_portfolio_value(start_date, end_date, myport, allocations, \
                                                        gen_plot=True)

    print(cr, adr, sddr, sr, ev)
