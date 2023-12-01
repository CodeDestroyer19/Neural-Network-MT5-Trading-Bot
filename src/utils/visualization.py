# visualization.py
"""
Module for visualizing data for the trading bot.

This module provides functions for visualizing historical price data and trade signals.

Author: Mike Kiwalabye
"""

import matplotlib.pyplot as plt
import pandas as pd

def plot_price_data(df: pd.DataFrame, title: str = 'Price Chart') -> None:
    """
    Plot the historical price data.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing historical price data.
    - title (str): The title of the plot.

    Returns:
    - None
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['close'], label='Close')
        
        # Add visualizations for other indicators, levels, and patterns
        # (Add more visualizations as needed)

        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        
        # Use plt.show(block=True) to make the plot blocking
        plt.show(block=True)
    except Exception as e:
        print(f"Error plotting trade signals: {str(e)}")

# ...

def plot_trade_signals(df: pd.DataFrame, title: str = 'Trade Signals') -> None:
    """
    Plot trade signals on the historical price chart.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing historical price data with trade signals.
    - title (str): The title of the plot.

    Returns:
    - None
    """
    try:    
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['close'], label='Close')
        
        # Plot trade signals
        buy_signals = df[df['signal'] == 'Buy']
        sell_signals = df[df['signal'] == 'Sell']

        plt.scatter(buy_signals.index, buy_signals['close'], color='green', marker='^', label='Buy Signal')
        plt.scatter(sell_signals.index, sell_signals['close'], color='red', marker='v', label='Sell Signal')
        
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        
        # Use plt.show(block=True) to make the plot blocking
        plt.show(block=True)
    except Exception as e:
        print(f"Error plotting trade signals: {str(e)}")
