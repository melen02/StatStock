from loadData.load_yfinance import load_raw_data
from calculations.correlation import export_correlation
from calculations.regression import run_regression
from calculations.monte_carlo import run_monte_carlo

# 0 = All
# 1 = Fundamentals
# 2 = CompanyQuarterlyData
# 3 = Stock_History
#UPDATE_MODE = 3
#load_raw_data(update_mode=UPDATE_MODE)
#run_regression()
#export_correlation()

RUN_MONTE_CARLO = True

if RUN_MONTE_CARLO:
    run_monte_carlo()