import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import utilities as util


def load_allocations():
    actual_allocations = util.load_csv_as_df('logs','actual', 'Symbol')
    target_allocations = util.load_csv_as_df('logs','target', 'Symbol')

    return actual_allocations, target_allocations


def create_orders(symbols=['AAPL', 'GOOG'], actual=[0.5,0.5], target=[0.6,0.4], trade_date=dt.date.today(),
                  max_trade_size=0.10, verbose=False, savelogs=False):
    #actual = pd.DataFrame(actual)
    #target = pd.DataFrame(target)
    #tempa, tempt = actual, target
    #actual, target = load_allocations()

    buy = []
    sell = []

    # First stab at creating orders, with a limit of 10% on biggest trade size
    for i in range(len(symbols)):

        if actual[i] > target[i]:
            delta = round(actual[i] - target[i], 3)
            delta = min(delta, max_trade_size)
            change = [trade_date.strftime("%m/%d/%Y"), symbols[i], 'SELL', delta]
            sell.append(change)
        elif actual[i] < target[i]:
            delta = round(target[i] - actual[i], 3)
            delta = min(delta, max_trade_size)
            change = [trade_date.strftime("%m/%d/%Y"), symbols[i], 'BUY', delta] #, columns=cols)
            buy.append(change)


    cols = ['Date', 'Symbol', 'Action', 'Quantity']
    ind = ['Date', 'Symbol']

    buy = pd.DataFrame(data=buy, columns=cols)
    #buy = buy.set_index(ind)
    sell = pd.DataFrame(sell, columns=cols)
    #sell = sell.set_index(ind)

    total_bought = buy['Quantity'].sum()
    total_sold = sell['Quantity'].sum()

    # Update Buy Orders in case not enough was sold
    if total_bought > total_sold:
        buy['Quantity'] *= total_sold/total_bought
        buy['Quantity'] = np.round(buy['Quantity'], 3)
        total_bought = buy['Quantity'].sum()
        buy.loc[0, 'Quantity'] += total_sold - total_bought

    # Update Sell Orders in case not enough was bought
    elif total_bought < total_sold:
        sell['Quantity'] *= total_bought/total_sold
        sell['Quantity'] = np.round(sell['Quantity'], 3)
        total_sold = sell['Quantity'].sum()
        sell['Quantity'][0] = sell['Quantity'][0] + total_bought - total_sold

    #try:
    #    assert 1 == 2, 'Inequals amounts bought or sold'
    #except AssertionError as e:
    #    print(e.message)

    #assert buy['Quantity'].sum() == sell['Quantity'].sum(), 'Inequal amounts bought or sold'

    orders = pd.concat([sell, buy], axis=0, ignore_index=True)
    orders['Quantity'] = np.round(orders['Quantity'], 3)


    if savelogs:

        all_orders = util.load_csv_as_df('logs', 'orders', 'Trade_Num')
        #all_orders.set_index(ind)
        all_orders = pd.concat([all_orders, orders], axis=0, ignore_index=True)

        util.save_df_as_csv(all_orders, 'logs', 'orders', 'Trade_Num')


    return orders


if __name__ == "__main__":
    print("This is just a training module.  Run portfolio-stats.py")

    orders = create_orders(0.10)

#    myport = ['VFFSX', 'VBTIX', 'LPSFX', 'VITPX', 'VMCPX', 'VSCPX', 'FMGEX', 'FSPNX']
#    allocations = [0.5668, 0.0453, 0.3879, 0.0, 0.0, 0.0, 0.0, 0.0]

    print(orders)

