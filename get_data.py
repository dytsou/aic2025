import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import os, sys
import argparse
from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor as YahooDownloader 
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl import config_tickers
from finrl.config import INDICATORS, DATA_SAVE_DIR

import itertools

START_DATE = '2024-01-01' 
END_DATE = '2024-05-15'
TICKER_LIST = config_tickers.DOW_30_TICKER 
FILE_PATH = 'trade_data.csv'
TIME_INTERVALS = ['1D', '1H'] 

def handle_args():
    parser = argparse.ArgumentParser(description='Fetch and preprocess stock data.')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('-s', '--start_date', default=START_DATE, type=str, help='Start date')
    parser.add_argument('-e', '--end_date', default=END_DATE, type=str, help='End date')
    parser.add_argument('-t', '--time_interval', default='1D', type=str, help='Time interval: 1D, 1H, etc.')
    parser.add_argument('-p', '--file_path', default=FILE_PATH, type=str, help='Data file')
    parser.add_argument('-L', '--ticker_list', default='CUSTOM', type=str, help='Ticker list name')
    parser.add_argument('-T', '--tickers', default=None, type=str, nargs='+', help='Specific tickers')
    return parser.parse_args()

def interact():
    print("Interactive mode: Enter data parameters")
    start_date = str(input(f"Start date ({START_DATE}): ")) or START_DATE
    end_date = str(input(f"End date ({END_DATE}): ")) or END_DATE
    file_path = str(input(f"Data file ({FILE_PATH}): ")) or FILE_PATH
    time_interval = str(input(f"Time interval (1D): ")) or '1D'
    ticker_list = str(input(f"Ticker list (CUSTOM): ")) or 'CUSTOM'
    tickers = input("Tickers (e.g., AAPL MSFT): ").split() or TICKER_LIST
    return argparse.Namespace(interactive=True, start_date=start_date, end_date=end_date, time_interval=time_interval, file_path=file_path, ticker_list=ticker_list, tickers=tickers)

class DataFetcher:
    def __init__(self, start_date, end_date, time_interval, ticker_list):
        self.start_date = start_date
        self.end_date = end_date
        self.time_interval = time_interval
        self.ticker_list = ticker_list
    
    def fetch_data(self):
        df_raw = YahooDownloader().download_data(self.ticker_list, self.start_date, self.end_date, time_interval=self.time_interval)
        if pd.api.types.is_string_dtype(df_raw['timestamp']):
            df_raw['date'] = pd.to_datetime(df_raw['timestamp'].str[:19], utc=False)
        else:
            df_raw['date'] = pd.to_datetime(df_raw['timestamp'], utc=False)
        df_raw = df_raw.drop(columns=['timestamp'])
        return df_raw
        
    def preprocess_data(self, df_raw):
        fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=INDICATORS, use_turbulence=False, user_defined_feature=False)
        processed = fe.preprocess_data(df_raw)
        list_ticker = processed["tic"].unique().tolist()
        list_date = processed["date"].unique().tolist()
        combination = list(itertools.product(list_date, list_ticker))
        processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
        processed_full = processed_full[processed_full['date'].isin(processed['date'])].sort_values(['date', 'tic']).fillna(0)
        return processed_full
        
    def get_data(self):
        df_raw = self.fetch_data()
        return self.preprocess_data(df_raw)

if __name__ == '__main__':
    args = handle_args()
    if args.interactive:
        args = interact()
    ticker_list = args.tickers if args.tickers else TICKER_LIST
    for interval in TIME_INTERVALS:
        df = DataFetcher(args.start_date, args.end_date, interval, ticker_list)
        data = df.get_data()
        file_path = os.path.join(DATA_SAVE_DIR, f'trade_data_{interval}.csv')
        if not os.path.exists(DATA_SAVE_DIR):
            os.makedirs(DATA_SAVE_DIR)
        data.to_csv(file_path)
        print(f"Data saved to {file_path} [{data.shape[0]} rows x {data.shape[1]} columns]")