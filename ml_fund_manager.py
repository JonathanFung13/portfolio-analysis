import forecaster as fc
import optimizer as opt
import trader as td
import datetime as dt
import utilities as util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calc_start_date(end_date=dt.datetime(2017,1,1), data_size=12):
    return end_date-dt.timedelta(weeks=int(data_size * 52/12))

def run_today(start_date=dt.datetime(2015,1,1), end_date=dt.datetime(2017,1,1), n_days=21, data_size=12,
              myport=['AAPL', 'GOOG'], allocations=[0.5,0.5],
              train_size=0.7, max_k=50, max_trade_size=0.1, gen_plot=False, verbose=False, savelogs=False):
    """



    :param start_date: Beginning of time period
    :param end_date: End of time period
    :param n_days: Number of days into the future to predict the daily returns of a fund
    :param data_size: The number of months of data to use in the machine learning model.
    :param myport: The funds available in your portfolio
    :param allocations: The percentage of your portfolio invested in the funds
    :param train_size: The percentage of data used for training the ML model, remained used for testing.
    :param max_k: Maximum number of neighbors used in kNN
    :param max_trade_size: The maximum percentage of your portfolio permitted to be traded in any one transaction.
    :param gen_plot: Boolean to see if you want to plot results
    :param verbose: Boolean to print out information during execution of application.
    :return:
    """


    start_date = calc_start_date(end_date, data_size)#end_date - dt.timedelta(weeks=int(data_size * 52/12))
    #print('start:', start_date, 'end:', end_date)

    if verbose: print('-'*20 + '\nFORECAST\n' + '-'*20)
    forecast = fc.forecast(start_date, end_date, symbols=myport, train_size=train_size,
                           n_days=n_days, max_k=max_k, gen_plot=gen_plot, verbose=verbose, savelogs=savelogs)

    if verbose: print('\n'+'-'*20 + '\nOPTIMIZE\n' + '-'*20)
    target_allocations = opt.optimize_return(forecast, myport, allocations, gen_plot=gen_plot, verbose=verbose, savelogs=savelogs)

    if verbose: print('\n' + '-'*20 + '\nORDERS\n' + '-'*20)
    trade_date = forecast.index.max()
    orders = td.create_orders(myport, allocations, target_allocations, trade_date=trade_date,max_trade_size=max_trade_size, verbose=verbose, savelogs=savelogs)

    if verbose: print(orders)

    new_allocations = allocations.copy()
    for i in range(orders.shape[0]):
        # fix this code so that the correct allocations are updated!
        index = myport.index(orders.loc[i, 'Symbol'])
        #symbol = orders.loc[i, 'Symbol']

        if orders.loc[i, 'Action'] == 'SELL':
            new_allocations[index] -= orders.loc[i, 'Quantity']
        else:
            new_allocations[index] += orders.loc[i, 'Quantity']


    adr_current, vol_current, sr_current, pv_current = util.compute_returns(forecast, allocations=allocations)
    adr_target, vol_target, sr_target, pv_target = util.compute_returns(forecast, allocations=target_allocations)
    adr_new, vol_new, sr_new, pv_new = util.compute_returns(forecast, allocations=new_allocations)

    if verbose:
        print("Portfolios:", "Current", "Target","New")
        print("Daily return: %.5f %.5f %.5f" % (adr_current, adr_target, adr_new))
        print("Daily Risk: %.5f %.5f %.5f" % (vol_current, vol_target, vol_new))
        print("Sharpe Ratio: %.5f %.5f %.5f" % (sr_current, sr_target, sr_new))
        print("Return vs Risk: %.5f %.5f %.5f" % (adr_current/vol_current, adr_target/vol_target, adr_new/vol_new))
        print("\nALLOCATIONS\n" + "-" * 40)
        print("Symbol", "Current", "Target", 'New')
        for i, symbol in enumerate(myport):
            print("%s %.3f %.3f %.3f" %
                  (symbol, allocations[i], target_allocations[i], new_allocations[i]))

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:

        fig, ax = plt.subplots()
        ax.scatter(vol_current, adr_current, c='green', s=15, alpha=0.5) # Current portfolio
        ax.scatter(vol_target, adr_target, c='red', s=15, alpha=0.5) # ef
        ax.scatter(vol_new, adr_new, c='black', s=25, alpha=0.75) # ef
        ax.set_xlabel('St. Dev. Daily Returns')
        ax.set_ylabel('Mean Daily Returns')
        #ax.set_xlim(min(vol)/1.5, max(vol)*1.5)
        #ax.set_ylim(min(adr)/1.5, max(adr)*1.5)
        ax.grid()
        ax.grid(linestyle=':')
        fig.tight_layout()
        plt.show()


        # add code to plot here
        df_temp = pd.concat([pv_current, pv_target, pv_new], keys=['Current', 'Target', 'New'], axis=1)
        df_temp = df_temp / df_temp.ix[0, :]
        util.plot_data(df_temp, 'Forecasted Daily portfolio value and SPY', 'Date-21', 'Normalized Price')

    if False: # meh was going to plot portfolio values for the last year but trying something else now
        prior_prices = util.load_data(myport, start_date, end_date)
        prior_prices.fillna(method='ffill', inplace=True)
        prior_prices.fillna(method='bfill', inplace=True)


        #prices_SPY = prior_prices['SPY']  # SPY prices, for benchmark comparison
        prior_prices = prior_prices[myport]  # prices of portfolio symbols

        forecast_prices = forecast * prior_prices

        time_span = pd.date_range(forecast.index.min(), end_date + dt.timedelta(days=n_days*2))
        forecast_prices = forecast_prices.reindex(time_span)
        forecast_prices = forecast_prices.shift(periods=n_days*2)
        forecast_prices = forecast_prices.dropna()

        forecast_prices = pd.concat([prior_prices, forecast_prices], axis=0)

        adr_current, vol_current, sr_current, pv_current = util.compute_returns(forecast_prices, allocations=allocations)
        adr_target, vol_target, sr_target, pv_target = util.compute_returns(forecast_prices, allocations=target_allocations)
        adr_new, vol_new, sr_new, pv_new = util.compute_returns(forecast_prices, allocations=new_allocations)

        df_temp = pd.concat([pv_current, pv_target, pv_new], keys=['Current', 'Target', 'New'], axis=1)
        df_temp = df_temp / df_temp.ix[0, :]
        util.plot_data(df_temp, 'Daily portfolio value and SPY', 'Date', 'Normalized Price')

    return new_allocations, trade_date

def test_experiment_one(n_days=21, data_size=12, train_size=0.7, max_k=50, max_trade_size=0.1,
                        years_to_go_back=2, initial_investment=10000, gen_plot=False, verbose=False, savelogs=False):

    today = dt.date.today()
    yr = today.year - years_to_go_back
    mo = today.month - 1  # Just temporary, take out 1 when data download is fixed.
    da = today.day - 1

    start_date = dt.datetime(yr, mo, da)
    end_date = dt.datetime(yr + 1, mo, da)

    adr = [None] * 12
    vol = [None] * 12
    sr = [None] * 12

    myport = ['AAPL', 'GLD']
    myalloc = [0.5,0.5]

    myport, myalloc = util.verify_allocations()

    # Portfolio values for Holding the Same Allocation (conservative case)
    actual_prices = util.load_data(myport, start_date, end_date)
    actual_prices.fillna(method='ffill', inplace=True)
    actual_prices.fillna(method='bfill', inplace=True)
    prices_SPY = actual_prices['SPY']
    actual_prices = actual_prices[myport]

    adr_cons, vol_cons, sharpe_cons, pv_cons = util.compute_returns(actual_prices, myalloc, sf=252.0, rfr=0.0)

    # Portfolio values with monthly optimization using hindsight (best possible case)


    # Portfolio values for Machine Learner

    ml_allocs = []
    ml_trade_dates = []

    for i in range(int(252/n_days)):
        temp = round(i*52*n_days/252)
        test_date = start_date + dt.timedelta(weeks=round(i*52*n_days/252))
        #print(i, temp, test_date)

        if verbose: print(('EXPERIMENT %i - %s') % (i, str(test_date.strftime("%m/%d/%Y"))))

        myalloc, trade_date = run_today(end_date=test_date, n_days=n_days, data_size=data_size,
                                    myport=myport, allocations=myalloc,
                                    train_size=train_size, max_k=max_k,
                                    max_trade_size=max_trade_size, gen_plot=gen_plot, verbose=verbose, savelogs=savelogs)

        ml_allocs.append(myalloc)
        ml_trade_dates.append(trade_date)


    ml_allocations = pd.DataFrame(data=ml_allocs, index=ml_trade_dates, columns=myport)
    all_dates = actual_prices.index
    #ml_allocaations = ml_allocaations.reindex(all_dates, method='ffill')

    actual_prices['Cash'] = 1.0

    ml_holdings = pd.DataFrame(data=0.0, index=all_dates, columns=myport)
    ml_holdings['Cash'] = 0.0
    ml_holdings.ix[0,'Cash'] = initial_investment
    values = ml_holdings * actual_prices
    porvals = values.sum(axis=1)

    for index, allocation in ml_allocations.iterrows():
        if index < ml_holdings.index.min():
            index = ml_holdings.index.min()
        #else:
        #    index = ml_holdings.index.get_loc(tdate, method='ffill')
        tomorrow = ml_holdings.index.get_loc(index) + 1

        for symbol in myport:
            ml_holdings.loc[tomorrow:, symbol] = porvals.loc[index] * allocation[symbol] / actual_prices.loc[index,symbol]

        values = ml_holdings * actual_prices
        porvals = values.sum(axis=1)


    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([pv_cons, porvals, prices_SPY], keys=['Conservative', 'ML', 'SPY'],
                            axis=1)
        df_temp = df_temp / df_temp.ix[0, :]
        util.plot_data(df_temp, 'Daily portfolio value and SPY', 'Date', 'Normalized Price')

    ret_cons = (pv_cons[-1] / pv_cons[0]) - 1
    ret_porvals = (porvals[-1] / porvals[0]) - 1
    ret_SPY = (prices_SPY[-1] / prices_SPY[0]) - 1

    return ret_cons, ret_porvals, ret_SPY

if __name__ == "__main__":

    test = True

    initial_investment = 10000 # dollars invested from start

    today = dt.date.today()
    yr = today.year
    mo = today.month  # Just temporary, take out 1 when data download is fixed.
    da = today.day - 1

    end_date = dt.datetime(yr, mo, da)

    if test:
        print("Running ML Fund Manager")

        myport, allocations = util.verify_allocations()

        n_days = 21 # How long the forecast should look out
        data_size = 12 # Number of months of data to use for Machine Learning
        train_size = 0.60 # Percentage of data used for training, rest is test
        max_k = 15 # Maximum value of k for kNN
        max_trade_size= 0.2 # Maximum amount of allocation allowed in a trade

        run_today(end_date=end_date, n_days=n_days, data_size=data_size,
                  myport=myport, allocations=allocations,
                  train_size=train_size, max_k=max_k,
                  max_trade_size=max_trade_size, gen_plot=True, verbose=True)
    else:
        print("Testing ML Fund Manager")

        n_days = 21  # How long the forecast should look out
        data_size = 12  # Number of months of data to use for Machine Learning
        train_size = 0.70  # Percentage of data used for training, rest is test
        max_k = 5  # Maximum value of k for kNN
        max_trade_size = 0.10  # Maximum amount of allocation allowed in a trade

        years_to_go_back = 3

        n_days = [14, 21]  # How long the forecast should look out
        data_size = [3, 6, 12] #, 18]  # Number of months of data to use for Machine Learning
        train_size = [0.6, 0.7, 0.8]  # Percentage of data used for training, rest is test
        max_k = [5, 10, 15] #, 20, 25]  # Maximum value of k for kNN
        max_trade_size = [0.10, 0.20, 0.30] #, 0.40]  # Maximum amount of allocation allowed in a trade
        years_to_go_back = [1]

        r_cons = []
        r_porvals = []
        r_SPY = []
        nd = []
        ds = []
        ts = []
        mk = []
        maxt = []
        yrs = []
        nums = []


        exp_no = 0


        for year in years_to_go_back:
            for mts in max_trade_size:
                for k in max_k:
                    for t in train_size:
                        for d in data_size:
                            for n in n_days:
                                #print(n, d, t, k, mts, year)

                                one, two, three = test_experiment_one(n_days=n, data_size=d, train_size=t, max_k=k,
                                                                      max_trade_size=mts, years_to_go_back=year,
                                                                      gen_plot=False, verbose=False, savelogs=False)
                                exp_no += 1

                                print(exp_no, one, two, three, n, d, t, k, mts, year)
                                nums.append(exp_no)
                                r_cons.append(one)
                                r_porvals.append(two)
                                r_SPY.append(three)
                                nd.append(n)
                                ds.append(d)
                                ts.append(t)
                                mk.append(k)
                                maxt.append(mts)
                                yrs.append(year)

        results = pd.DataFrame(data=list(zip(r_cons, r_porvals, r_SPY, nd, ds, ts, mk, maxt, yrs)),
                               columns=['cons_return', 'ml_return', 'spy_return', 'forecast', 'months_of_data',
                                        'train_size', 'maxk', 'max_trade', 'yrs_lookback'],
                               index=nums)
        util.save_df_as_csv(results,'results','%s_results' % (dt.date.today().strftime("%Y_%m_%d")), indexname='exp_num')


                        #test_experiment_one(n_days=n_days, data_size=data_size, train_size=train_size, max_k=max_k,
        #                    max_trade_size=max_trade_size, years_to_go_back=years_to_go_back, gen_plot=False)

