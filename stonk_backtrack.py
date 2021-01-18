#stonk_backtrack


# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# SUP BRO
import pandas as pd
import bs4 as bs
import os
import pickle
import requests
import datetime as dt
import yfinance as finance
import pandas_datareader.data as web
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn import svm, neighbors
from sklearn.model_selection import cross_validate
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

style.use('ggplot')

import matplotlib.dates as mdates
import numpy as np


# def save_sp500_tickers():
#     resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
#     soup = bs.BeautifulSoup(resp.text)
#     table = soup.find('table', {'class': 'wikitable sortable'})
#     tickers = []
    
#     for row in table.findAll('tr')[1:]:
#         ticker = row.findAll('td')[0].text.strip()

#         tickers.append(ticker)
    
    

#     for row in table.findAll('tr')[1:]:
#         ticker = row.findAll('td')[0].text
#         tickers.append(ticker)

#     tickers1 = []
#     for ticker in tickers:
#         replaced = ticker.replace(".", "-")
#         tickers1.append(replaced)



#     with open("sp500tickers.pickle", "wb") as f:
#         pickle.dump(tickers1, f)
    
#     return tickers1

#save_sp500_tickers()



#ticker_watch_list = ['AAPL', 'TSLA', 'AMZN', 'MU', 'TSM', 'PFE', 'QQQJ', 'MJ', 'XL', 'SQQQ', 'TQQQ', 'DDOG', 'XOM']
#pass list of stocks you want to find then loop through it 
def get_data(ticker, start_date, end_date):
    if not os.path.exists('stock_csvs'):
        os.makedirs('stock_csvs')
    
    start = start_date
    end = end_date



    try:
    #for ticker in tickers
        df = web.DataReader(ticker, 'yahoo', start, end)
        df.to_csv('stock_csvs/{}.csv'.format(ticker))
        return 0
    except:
        print(ticker + " is not a Valid Stock Ticker")
        return 1


#Works for all date combos including future and weekends.
#MAKE AUTOMATED! every day     
#get_data(ticker_watch_list, dt.datetime(2020,1,4), dt.datetime(2021,1,10))


#all_stocks = pd.DataFrame()
# def compile_data():
#     with open ("sp500tickers.pickle", "rb") as f:
#         tickers = pickle.load(f)
#     main_df = pd.DataFrame()
#     for count,ticker in enumerate(tickers[:500]):
        
#         df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
#         df.set_index('Date', inplace = True)
#         df.rename(columns = {'Adj Close': ticker}, inplace = True)
#         df.drop(['Open','High','Low','Close','Volume'], 1, inplace = True)
#         if main_df.empty:
#             main_df = df
#         else:
#             main_df = main_df.join(df, how='outer')
#     all_stocks = main_df
#     print(main_df.head())
#     main_df.to_csv('sp500_joined_closes.csv')
 
#compile_data()


# %%
import yfinance as yf
import datetime as dt
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import datetime as datetime
import numpy as np
import mplfinance as mpf
from IPython.display import Image

moving_averages = [4, 10, 30]

every_stock = {}
def make_data(ticker_watch_list):

    for stock in ticker_watch_list: 
        # Adding moving averages to the dataframe
        stock_df = pd.read_csv('stock_csvs/' + stock + '.csv', parse_dates=True, index_col=0)
        stock_df['MA-4'] = stock_df['Adj Close'].rolling(window=4, min_periods=0).mean()
        stock_df['MA-10'] = stock_df['Adj Close'].rolling(window=10, min_periods=0).mean()
        stock_df['MA-30'] = stock_df['Adj Close'].rolling(window=30, min_periods=0).mean()

        # Creating Bollinger Bands
        stock_df['MA-15'] = stock_df['Adj Close'].rolling(window=15, min_periods=0).mean()
        stock_df['STDEV'] = stock_df.iloc[:,5].rolling(window = 15, min_periods=0).std()
        stock_df['Lower Band'] = stock_df['MA-15'] - (2 * stock_df['STDEV'])
        stock_df['Upper Band'] = stock_df['MA-15'] + (2 * stock_df['STDEV'])
        stock_df["Date"] = mdates.date2num(stock_df.index)

    # Creating the 10.4 and 10.4.4 stochastic
        Period = 10
        K = 4
        stock_df['RolHigh'] = stock_df['High'].rolling(window=Period, min_periods=0).max()
        stock_df['RolLow'] = stock_df['Low'].rolling(window=Period, min_periods=0).min()
        stock_df['Stochastic'] = ((stock_df['Adj Close'] - stock_df['RolLow']) / (stock_df['RolHigh'] - stock_df['RolLow'])) * 100
        stock_df['Fast Stochastic'] = stock_df['Stochastic'].rolling(window=K, min_periods=0).mean()
        stock_df['Slow Stochastic'] = stock_df['Fast Stochastic'].rolling(window=K, min_periods=0).mean()



        # Creating Columns for RWB Charts
        emasUsed=[3,5,8,10,12,15,30,35,40,45,50,60]
        for x in emasUsed:
            ema=x
            stock_df["Ema_"+str(ema)]=round(stock_df.iloc[:,5].ewm(span=ema, adjust=False).mean(),2)
        
        stock_df['Color Chart'] = 'B'

        for i in stock_df.index:
            cmin=min(stock_df["Ema_3"][i],stock_df["Ema_5"][i],stock_df["Ema_8"][i],stock_df["Ema_10"][i],stock_df["Ema_12"][i],stock_df["Ema_15"][i])
            cmax=max(stock_df["Ema_30"][i],stock_df["Ema_35"][i],stock_df["Ema_40"][i],stock_df["Ema_45"][i],stock_df["Ema_50"][i],stock_df["Ema_60"][i])
            if(cmin > cmax):
                stock_df['Color Chart'][i] = 'RWB'
            elif(cmin <= cmax):
                stock_df['Color Chart'][i] = 'BWR'



        # Making Green Dot check columns
        greenDotDate=[] 
        greenDot=[] 
        prevFast=0 
        prevSlow=0 
        lastLow=0 
        lastClose=0 
        lastLowBB=0 

        # A green dot indicator will appear when the fast stochastic is above the slow stochastic, the previous day has this same behavior, and if the previous day's fast stochastic is below 60.
        stock_df['Green Dot?'] = np.nan

        for i in stock_df.index:
            if stock_df['Fast Stochastic'][i] > stock_df['Slow Stochastic'][i] and prevFast < prevSlow and prevFast < 60:
                stock_df['Green Dot?'][i] = stock_df['Lower Band'][i]
    

            prevFast=stock_df['Fast Stochastic'][i]
            prevSlow=stock_df['Slow Stochastic'][i]
            lastLow=stock_df['Low'][i]
            lastClose=stock_df['Adj Close'][i]
            lastLowBB=stock_df['Lower Band'][i]


        

        # Making GMI indicator column
        stock_df['GMI'] = 'G'
        green_dates = ['2019-02-01', '2019-06-11', '2019-09-05', '2019-10-16']
        red_dates = ['2019-01-01', '2019-05-10', '2019-08-02', '2019-10-01']
        green_dates = pd.to_datetime(green_dates)
        red_dates = pd.to_datetime(red_dates)

        green = False
        for date in stock_df.index: 
            if date in green_dates:
                green = True
            if date in red_dates:
                green = False
            if green: 
                stock_df['GMI'][date] = 'Green'
            else:
                stock_df['GMI'][date] = 'Red'
            

        every_stock[stock] = stock_df
        #return every_stock
        

    #pd.set_option('display.max_columns', 30)
    # Displaying an example of what the finished dataframe will look like for each stock
    #every_stock['TSLA']



# %%
# ap0 = [ 
#         mpf.make_addplot(every_stock['AAPL']['Upper Band'],color='grey'),  
#         mpf.make_addplot(every_stock['AAPL']['Lower Band'],color='grey'),  
#         mpf.make_addplot(every_stock['AAPL']['MA-4'],color='red'),
#         mpf.make_addplot(every_stock['AAPL']['MA-10'],color='green'),
#         mpf.make_addplot(every_stock['AAPL']['MA-30'],color='blue'),
#         mpf.make_addplot(every_stock['AAPL']['Fast Stochastic'],color='r',panel=2),  
#         mpf.make_addplot(every_stock['AAPL']['Slow Stochastic'],color='b',panel=2),
#         mpf.make_addplot(every_stock['AAPL']['Green Dot?'],type='scatter', color='g', markersize=5)   
#       ]
      
# mpf.plot(every_stock['AAPL'],title='AAPL',type='candle', style='charles',volume=True,addplot=ap0,scale_width_adjustment=dict(ohlc=2.0,lines=0.4))

# %%
import datetime

def buy_sell(total_fund, stock, date, num_shares, num_stocks_held, stocks_owned, shares_bought, max_stocks, endDate):
    date_time_str = '2021-01-04 00:00:00'
    date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
    df = every_stock[stock]
    if df['Color Chart'][date] == 'RWB' and df['Green Dot?'][date] == df['Lower Band'][date] and not (stock in stocks_owned) and total_fund != 0 and len(stocks_owned) < max_stocks and date != date_time_obj:
        print("\nBuying " + str(stock) + " at: " + str(date) + "\n" + str(num_shares) + " shares\n" + str(df['Open'][date]) + " per share")
        return 2
    if df['Color Chart'][date] == 'BWR' and (stock in stocks_owned):
        print("\nSelling " + str(stock) + " becasue of BWR: " + str(date)  + "\n" + str(shares_bought[stocks_owned.index(str(stock))]) + " shares\n" + str(df['Adj Close'][date]) + " per share")
        return 1
    if date == date_time_obj and (stock in stocks_owned):
        print("\nSelling " + str(stock) + " at end of year at " + str(df['Adj Close'][date]) + " per share" + " for " + str((shares_bought[stocks_owned.index(str(stock))] * df['Adj Close'][date])))
        return 3
    else: 
        return 0


def trade(portfolio, original_p, date, df, stocks_owned, shares_bought, num_stocks_held, max_stocks, ticker, endDate):
    num_stocks_held = len(stocks_owned) 
    if (((original_p / max_stocks) / (df['Open'][date])) > 0.0): 
        if portfolio > (original_p / max_stocks):
            num_shares = ((original_p / max_stocks) / df['Open'][date]) 
        else:
            num_shares = portfolio / df['Open'][date]
    
    buy_sell_hold = buy_sell(portfolio, ticker, date, num_shares, num_stocks_held, stocks_owned, shares_bought, max_stocks, endDate)

    if buy_sell_hold == 2 and len(stocks_owned) <= max_stocks:
        print("Portfolio Value before buying " + str(ticker) + " = " + str(portfolio))
        stocks_owned.append(ticker)
        shares_bought.append(num_shares)
        s = (shares_bought[stocks_owned.index(str(ticker))])
        portfolio = portfolio - (s * df['Open'][date])
        print("Portfolio Value after buying " + str(ticker) + " = " + str(portfolio))
        return portfolio
    elif buy_sell_hold == 1 and (ticker in stocks_owned):
        portfolio = portfolio + (shares_bought[stocks_owned.index(str(ticker))] * df['Adj Close'][date])
        shares_bought.remove(shares_bought[stocks_owned.index(str(ticker))])
        stocks_owned.remove(str(ticker))
        print("Portfolio value = " + str(portfolio))
        return portfolio
    elif buy_sell_hold == 3 and (ticker in stocks_owned):
        portfolio = portfolio + (shares_bought[stocks_owned.index(str(ticker))] * df['Adj Close'][date])
        shares_bought.remove(shares_bought[stocks_owned.index(str(ticker))])
        stocks_owned.remove(str(ticker))
        print("Portfolio value = " + str(portfolio))
        return portfolio
    else:
        return portfolio


def run_simulation(portfolio, max_stocks, startDate, endDate, every_stock):
    original_p = portfolio
    stocks_owned = []
    shares_bought = []
    stocks_processed = []
    num_stocks_held = 0
    dates = []
   

    df = every_stock['TSLA']
    for date in df.index:
        dates.append(date)

    for date in dates:
        for stock in every_stock:
            if date >= every_stock[stock].index[0]:
                ticker = stock
                portfolio = trade(portfolio, original_p, date, every_stock[stock], stocks_owned, shares_bought, num_stocks_held, max_stocks, ticker, endDate)

    print("\n")
    print("Final portfolio value: " + str(portfolio))
    if original_p > portfolio:
        print("Percent made in the year: " + "-" + str((1 - (portfolio / original_p)) * 100) + "%")
    else:
        print("Percent made in the year: " + str(100 * ((portfolio / original_p) -1)) + "%")

# %%
def run_daily(tickers, date):

    dayno = date.weekday()
    if(dayno==5):
        date = date + dt.timedelta(days = -1)
    if(dayno==6):
        date = date + dt.timedelta(days = -2)
    print("As of Market Close On: " + date.strftime("%B %d, %Y"))
    for ticker in tickers:
        buy_sell(portfolio, ticker, date, num_shares, num_stocks_held, stocks_owned, shares_bought, max_stocks)

# %%
#date = '2021-1-6 00:00:00'
#print('On ' + str(date))
#run_daily(ticker_watch_list, date)
# %%
import warnings

warnings.filterwarnings("ignore")



    


def main():

    import datetime as dt
    from datetime import datetime
    ticker_watch_list = []
    startDate = input("Enter Start Date (Y-M-D): ")
    endDate = input("Enter End Date (Y-M-D): ")
    startDate = dt.datetime.strptime(startDate, '%Y-%m-%d')
    endDate = dt.datetime.strptime(endDate, '%Y-%m-%d')
    amount = int(input("Enter Starting Portfolio Value: "))
    
    txt_bool = input("Use Text File as Watchlist> (Y/N): ")
    if(txt_bool=='Y' or txt_bool == 'y'):
        file_name = input("Enter Text File: ")
        with open(file_name) as f:
            watch = f.read().splitlines()
        for line in watch:

            if(len(line)>=1):
                res = get_data(line, startDate, endDate)
                if(res==0):
                    ticker_watch_list.append(line)
            else:
                break
    else:
        print("Enter Stocks to Track (Enter 0 to Stop):")
        while True:
            inp = input("Enter Ticker: ")
            if(inp=='0'):
                print("Current watch list: "+ str(ticker_watch_list))
                break
            else:
                res = get_data(inp, startDate, endDate)
                if(res==0):
                    ticker_watch_list.append(inp)

    #get_data(ticker_watch_list, StartDate, curDate)
    make_data(ticker_watch_list)
    
    run_simulation(amount, 6, startDate, endDate, every_stock)
    
     
    print("Done!")   


    
    
    

if __name__ == "__main__":
    main()






# %% Percentage buy/sell/hold s and p?