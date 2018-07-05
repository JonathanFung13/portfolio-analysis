import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import utilities as util


def load_allocations():
    actual_allocations = util.load_csv_as_df('allocations','actual',None)
    target_allocations = util.load_csv_as_df('allocations','target',None)

    return actual_allocations, target_allocations


def create_orders(max_trade_size=0.10):
    actual, target = load_allocations()
    cols = ['Date', 'Symbol', 'Action', 'Quantity']

    orders = [] #pd.DataFrame(columns=['Date', 'Symbol', 'Action', 'Quantity'])
    buy = []
    sell = []

    # First stab at creating orders, with a limit of 10% on biggest trade size
    for i in range(actual.shape[0]):

        if actual.ix[i][1] > target.ix[i][1]:
            delta = round(actual.ix[i][1] - target.ix[i][1], 3)
            delta = min(delta, max_trade_size)
            change = [dt.date.today().strftime("%m/%d/%Y"), actual.ix[i][0], 'SELL', delta]
            sell.append(change)
        elif actual.ix[i][1] < target.ix[i][1]:
            delta = round(target.ix[i][1] - actual.ix[i][1], 3)
            delta = min(delta, max_trade_size)
            change = [dt.date.today().strftime("%m/%d/%Y"), actual.ix[i][0], 'BUY', delta] #, columns=cols)
            buy.append(change)

    buy = pd.DataFrame(buy, columns=cols)
    sell = pd.DataFrame(sell, columns=cols)

    total_bought = buy['Quantity'].sum()
    total_sold = sell['Quantity'].sum()

    # Update Buy Orders in case not enough was sold
    if total_bought > total_sold:
        buy['Quantity'] *= total_sold/total_bought
    # Update Sell Orders in case not enough was bought
    elif total_bought < total_sold:
        sell['Quantity'] *= total_bought/total_sold

    orders = pd.concat([buy, sell], axis=0)
    orders['Quantity'] = np.round(orders['Quantity'], 3)

    return orders


if __name__ == "__main__":
    print("This is just a training module.  Run portfolio-stats.py")

    orders = create_orders(0.10)

#    myport = ['VFFSX', 'VBTIX', 'LPSFX', 'VITPX', 'VMCPX', 'VSCPX', 'FMGEX', 'FSPNX']
#    allocations = [0.5668, 0.0453, 0.3879, 0.0, 0.0, 0.0, 0.0, 0.0]

    print(orders)
