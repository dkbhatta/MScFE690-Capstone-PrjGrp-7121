import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import time

import yfinance as yf
from ta import add_all_ta_features
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator


## Stock Data Download
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)[['Open', 'High', 'Low', 'Close', 'Volume']]
    data['Ticker'] = ticker
    return data

def add_technical_indicators(df):
    try:
        df = add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume"
        )
    except ValueError as e:
        print(f"Error applying technical indicators: {e}")
        # Returns the original df if indicators cannot be applied
        return df  
    return df

## Preprocess and Implementation
def fill_nan_values(df):
    # Group by ticker to handle each stock separately
    return df.groupby('Ticker').apply(lambda x: x.fillna(method='bfill').fillna(method='ffill'))

def add_bb_rsi(df, bb_window=20, bb_std=2, rsi_window=14):
    # Calculate Bollinger Bands
    bb_indicator = BollingerBands(close=df['Close'], window=bb_window, window_dev=bb_std)
    df['bb_high'] = bb_indicator.bollinger_hband()
    df['bb_low'] = bb_indicator.bollinger_lband()
    df['bb_mid'] = bb_indicator.bollinger_mavg()
    
    # Calculate RSI
    rsi_indicator = RSIIndicator(close=df['Close'], window=rsi_window)
    df['rsi'] = rsi_indicator.rsi()
    
    return df

def calculate_true_range(df):
    df['TR'] = np.maximum(df['High'] - df['Low'], 
                          np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                     abs(df['Low'] - df['Close'].shift(1))))
    return df

def coiled_spring_nr7(df):
    df['ATR'] = df['TR'].rolling(window=7).mean()
    df['NR7'] = df['TR'].rolling(window=7).min() == df['TR']
    
    df['long_signal'] = (df['NR7'] & (df['Close'] > df['bb_mid']) & (df['rsi'] < 50))
    df['short_signal'] = (df['NR7'] & (df['Close'] < df['bb_mid']) & (df['rsi'] > 50))
    
    return df

def finger_finder(df, atr_period=14, atr_multiplier=2):
    df['ATR'] = df['TR'].rolling(window=atr_period).mean()
    df['Upper_Band'] = df['High'].rolling(window=2).max() + (df['ATR'] * atr_multiplier)
    df['Lower_Band'] = df['Low'].rolling(window=2).min() - (df['ATR'] * atr_multiplier)
    
    df['long_signal'] = (df['Close'] > df['Upper_Band'].shift(1)) & (df['rsi'] < 70)
    df['short_signal'] = (df['Close'] < df['Lower_Band'].shift(1)) & (df['rsi'] > 30)
    
    return df

def power_spike(df, volume_threshold=2):
    df['volume_ma'] = df['Volume'].rolling(window=20).mean()
    df['price_change'] = df['Close'] - df['Open']
    
    df['long_signal'] = (df['Volume'] > df['volume_ma'] * volume_threshold) & (df['price_change'] > 0) & (df['rsi'] < 70)
    df['short_signal'] = (df['Volume'] > df['volume_ma'] * volume_threshold) & (df['price_change'] < 0) & (df['rsi'] > 30)
    
    return df

def backtest_strategy(df, initial_capital, transaction_cost_pct=0.001, risk_reward_ratio=2, trailing_stop_pct=0.05):
    df['position'] = np.where(df['long_signal'], 1, np.where(df['short_signal'], -1, 0))
    df['position'] = df['position'].fillna(method='ffill')
    
    df['entry_price'] = np.nan
    df['stop_loss'] = np.nan
    df['take_profit'] = np.nan
    df['trailing_stop'] = np.nan
    
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    trailing_stop = 0
    
    for i in range(1, len(df)):
        if df['position'].iloc[i] != 0 and position == 0:
            position = df['position'].iloc[i]
            entry_price = df['Close'].iloc[i]
            stop_loss = entry_price * (1 - position * trailing_stop_pct)
            take_profit = entry_price * (1 + position * trailing_stop_pct * risk_reward_ratio)
            trailing_stop = stop_loss if position == 1 else take_profit
        
        elif position != 0:
            if position == 1:
                trailing_stop = max(trailing_stop, df['Close'].iloc[i] * (1 - trailing_stop_pct))
            else:
                trailing_stop = min(trailing_stop, df['Close'].iloc[i] * (1 + trailing_stop_pct))
            
            if (position == 1 and df['Low'].iloc[i] <= trailing_stop) or \
               (position == -1 and df['High'].iloc[i] >= trailing_stop) or \
               df['position'].iloc[i] == -position:
                position = 0
                entry_price = 0
                stop_loss = 0
                take_profit = 0
                trailing_stop = 0
        
        df['entry_price'].iloc[i] = entry_price
        df['stop_loss'].iloc[i] = stop_loss
        df['take_profit'].iloc[i] = take_profit
        df['trailing_stop'].iloc[i] = trailing_stop
    
    df['returns'] = df['Close'].pct_change()
    df['strategy_returns'] = df['position'].shift(1) * df['returns'] - abs(df['position'].diff()) * transaction_cost_pct
    
    df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
    df['cumulative_strategy'] = initial_capital * df['cumulative_returns']
    
    total_return = df['cumulative_returns'].iloc[-1] - 1
    sharpe_ratio = np.sqrt(252) * df['strategy_returns'].mean() / df['strategy_returns'].std()
    max_drawdown = (df['cumulative_strategy'] / df['cumulative_strategy'].cummax() - 1).min()
    
    df['entry_long'] = (df['position'] == 1) & (df['position'].shift(1) != 1)
    df['entry_short'] = (df['position'] == -1) & (df['position'].shift(1) != -1)
    df['exit'] = (df['position'] == 0) & (df['position'].shift(1) != 0)
    
    return {
        'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Final Portfolio Value': df['cumulative_strategy'].iloc[-1]
    }


def process_stock(df, strategy_func, capital):
    df = add_bb_rsi(df)
    df = calculate_true_range(df)
    df = fill_nan_values(df)
    df = strategy_func(df)
    results = backtest_strategy(df, initial_capital=capital) 
    return results, df

## Data Stream
def stream_data(data):
    for word in data.split(" "):
        yield word + " "
        time.sleep(0.02)

def filter_dataframes_and_plot(selected_strategy, dataframes, sorted_profitable_stocks):
    filtered_dataframes = {}

    for ticker in sorted_profitable_stocks['Ticker']:
        if ticker in dataframes[selected_strategy]:
            # Extract the dataframe for the current ticker
            df = dataframes[selected_strategy][ticker]
            filtered_dataframes[ticker] = df

            # Create and display plot
            st.subheader(f"{ticker} - {selected_strategy} Entry and Exit Points")
            plt.figure(figsize=(15, 7))
            plt.plot(df.index, df['Close'], label='Close Price', alpha=0.5)
            plt.title(f'{ticker} Entry and Exit Points')
            plt.xlabel('Date')
            plt.ylabel('Price')

            # Plot entry and exit points
            plt.scatter(df[df['entry_long']].index, df[df['entry_long']]['Close'],
                        color='g', label='Entry Long', marker='^', s=100)
            plt.scatter(df[df['entry_short']].index, df[df['entry_short']]['Close'],
                        color='r', label='Entry Short', marker='v', s=100)
            plt.scatter(df[df['exit']].index, df[df['exit']]['Close'],
                        color='black', label='Exit', marker='x', s=100)

            plt.legend()
            plt.grid()
            plt.tight_layout()

            st.pyplot()
            st.divider() 

