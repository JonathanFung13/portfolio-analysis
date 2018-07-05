import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import utilities as util


def load_allocations():
    actual_allocations = util.load_csv_as_df('allocations','actual',None)
    target_allocations = util.load_csv_as_df('allocations','target',None)

    return actual_allocations, target_allocations


def create_orders():
    actual, target = load_allocations()

    print(target)

    return 1


if __name__ == "__main__":
    print("This is just a training module.  Run portfolio-stats.py")

    orders = create_orders()

#    myport = ['VFFSX', 'VBTIX', 'LPSFX', 'VITPX', 'VMCPX', 'VSCPX', 'FMGEX', 'FSPNX']
#    allocations = [0.5668, 0.0453, 0.3879, 0.0, 0.0, 0.0, 0.0, 0.0]

    print(orders)

