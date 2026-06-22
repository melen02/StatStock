from loadData.load_yfinance import load_raw_data
from calculations.correlation import export_correlation

# 0 = All
# 1 = Fundamentals
# 2 = CompanyQuarterlyData
# 3 = Stock_History
UPDATE_MODE = 3
load_raw_data(update_mode=UPDATE_MODE)
#export_correlation()