import yfinance as yf
import pandas as pd
import datetime


class Loader:
    def __init__(self, djia_year):
        self.djia_year = djia_year
        file = open(f'env/data/DJIA_{djia_year}/tickers.txt', 'r')
        self.tickers = [line.strip() for line in file.readlines()]
        self.stocks = []
        self.start_date = datetime.datetime(2018, 5, 20)
        self.end_date = datetime.datetime(2023, 5, 20)

    def download_data(self, start_date , end_date):
        for ticker in self.tickers:
            print(ticker)
            data = yf.download(ticker, start=start_date, end=end_date)
            data['ticker'] = ticker
            self.stocks.append(data)
            data.to_csv(f'env/data/DJIA_{self.djia_year}/ticker_{ticker}.csv')

    def read_data(self):
        for ticker in self.tickers:
            data = pd.read_csv(f'env/data/DJIA_{self.djia_year}/ticker_{ticker}.csv', parse_dates=True, index_col='Date')
            self.stocks.append(data)

    def load (self, start_date, end_date, download= False):
        if download:
            self.download_data(start_date, end_date)
        else:
            self.read_data()
        return self.stocks
