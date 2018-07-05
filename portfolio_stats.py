
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import scipy.optimize as spo
import utilities as util



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

    prices = util.load_data(symbols, start, end)
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)

    if gen_plot:
        # add code to plot here
#        df_temp = pd.concat([port_val, port_val_opt, port_val_ef, prices_SPY], keys=['Portfolio', 'Optimized', 'EF','SPY'], axis=1)
#        df_temp = df_temp / df_temp.ix[0, :]
        prices_funds = prices / prices.ix[0, :]

        plot_data(prices_funds, 'Daily Prices and SPY', 'Date', 'Normalized Price')

    prices_SPY = prices['SPY']  # SPY prices, for benchmark comparison
    prices = prices[symbols]  # prices of portfolio symbols



    # Get portfolio statistics
    cr, adr, sddr, sr, port_val = compute_returns(prices, allocations=allocations, rfr=rfr, sf=sf)

    optimized_allocations = optimal_allocations(prices)
    cr_opt, adr_opt, sddr_opt, sr_opt, port_val_opt = compute_returns(prices, allocations=optimized_allocations, rfr=rfr, sf=sf)

    # print("ALLOCATIONS\n" + "-" * 40)
    # for i, symbol in enumerate(symbols):
    #     print("%s actual: %.2f optimal: %.2f" % (symbol, allocations[i], optimized_allocations[i]))



    # Calculate efficient frontier
    num_portfolios = 500
    returns = [None] * num_portfolios
    risk = [None] * num_portfolios
    iallocations = [None] * num_portfolios
    risk_at_max = 0.0
    max_return = -10000.0


    for i in range(num_portfolios):
        weights = np.random.rand(len(symbols))
        iallocations[i] = weights / sum(weights)
        cr_i, returns[i], risk[i], sr_i, port_val_i = compute_returns(prices, allocations=iallocations[i], rfr=rfr, sf=sf)

        if returns[i] > max_return:
            max_return = returns[i]
            risk_at_max = risk[i]
            return_ef = returns[i]
            risk_ef = risk[i]
            allocations_ef = iallocations[i]

    risk_ef = risk_at_max

    for i, ireturn in enumerate(returns):
        if ireturn > (0.9 * max_return) and risk[i] < risk_ef:
            return_ef = returns[i]
            risk_ef = risk[i]
            allocations_ef = iallocations[i]

    cr_ef, adr_ef, sddr_ef, sr_ef, port_val_ef = compute_returns(prices, allocations=allocations_ef, rfr=rfr, sf=sf)

    print("Portfolios:", "Current", "Optimal", "Efficient")
    print("Cumul. return: %.5f %.5f %.5f" % (cr, cr_opt, cr_ef))
    print("Daily return: %.5f %.5f %.5f" % (adr, adr_opt, adr_ef))
    print("Daily Risk: %.5f %.5f %.5f" % (sddr, sddr_opt, sddr_ef))
    print("Return vs Risk: %.5f %.5f %.5f" % (adr/sddr, adr_opt/sddr_opt, adr_ef/sddr_ef))
    print("\nALLOCATIONS\n" + "-" * 40)
    print("", "Current", "Optimal", "Efficient")
    for i, symbol in enumerate(symbols):
        print("%s %.2f %.2f %.2f" %
              (symbol, allocations[i], optimized_allocations[i], allocations_ef[i]))

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:

        fig, ax = plt.subplots()
        ax.scatter(risk, returns, c='blue', s=5, alpha=0.2)
        ax.scatter(sddr, adr, c='green', s=8) # Current portfolio
        ax.scatter(sddr_opt, adr_opt, c='orange', s=8) # spo optimized
        ax.scatter(risk_at_max, max_return, c='red', s=8) # ef
        ax.set_xlabel('St. Dev. Daily Returns')
        ax.set_ylabel('Mean Daily Returns')
        ax.set_xlim(min(risk)/1.5, max(risk)*1.5)
        ax.set_ylim(min(returns)/1.5, max(returns)*1.5)
        ax.grid()
        ax.grid(linestyle=':')
        fig.tight_layout()
        plt.show()

        # plt.plot(risk, returns, 'o', markersize=5)
        # plt.plot(sddr, adr, 'g+') # Current portfolio
        # plt.plot(sddr_opt, adr_opt, 'b+') # spo optimized
        # plt.plot(risk_at_max, max_return, 'r+') # ef

        # add code to plot here
        df_temp = pd.concat([port_val, port_val_opt, port_val_ef, prices_SPY], keys=['Portfolio', 'Optimized', 'EF','SPY'], axis=1)
        df_temp = df_temp / df_temp.ix[0, :]
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

    if True:
        month = 1
        year = 2017
        start_date = dt.datetime(year, 6, 26)
        end_date = dt.datetime(year+1, 6, 26)

        target_retirement = [] #['LIKKX']
        passive_funds = ['DIS', 'TSLA','VFFSX', 'VITPX', 'VMCPX', 'VSCPX', 'MDDCX','FSPNX', 'VBTIX']
        active_funds = ['FMGEX', 'VMRXX', 'VUSXX']
        symbols = passive_funds + active_funds

        myport = ['VFFSX', 'VBTIX', 'BTCLP40', 'VITPX', 'VMCPX', 'VSCPX']
        allocations = [0.5668, 0.0453, 0.3879, 0.0, 0.0, 0.0]
        myport = ['VFFSX', 'VBTIX']
        allocations = [0.95, 0.05]
        myport = ['VFFSX', 'VBTIX', 'LPSFX', 'VITPX', 'VMCPX', 'VSCPX', 'FMGEX', 'FSPNX']
        allocations = [0.5668, 0.0453, 0.3879, 0.0, 0.0, 0.0, 0.0, 0.0]

        #myport = ['VFFSX', 'BTCLP40', 'VBTIX']
        #allocations = [0.55, 0.40, 0.05]

    #    prices = read_data(symbols, start_date, end_date)

        # Assess the portfolio
        cr, adr, sddr, sr, ev = compute_portfolio_value(start_date, end_date, myport, allocations, \
                                                            gen_plot=True)

        print(cr)
    else:
        start_date = dt.datetime(2005, 1, 1)
        end_date = dt.datetime(2018, 6, 14)

        myport = ['IBM', 'GLD', 'XOM', 'AAPL', 'MSFT', 'TLT', 'SHY']
        allocations = [0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.02]
        myport = ['IBM', 'GLD']
        allocations = [0.25, 0.75]

        # Assess the portfolio
        cr, adr, sddr, sr, ev = compute_portfolio_value(start_date, end_date, myport, allocations, \
                                                        gen_plot=True)

        print(cr)

