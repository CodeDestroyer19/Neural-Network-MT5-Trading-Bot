# trading_strategy.py
"""
Module for defining the trading strategy used by the trading bot.

This module provides functions for generating trade signals based on various indicators,
divergences, patterns, and trend directions.

Author: Mike Kiwalabye
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from talib import abstract
from src.models import neural_network_model

def get_historical_data(symbol: str, existing_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Retrieve historical data for a given symbol and timeframe from MetaTrader 5.

    Parameters:
    - symbol (str): The financial instrument symbol (e.g., 'EURUSD').
    - existing_data (pd.DataFrame): Existing historical data DataFrame.

    Returns:
    - pd.DataFrame: DataFrame containing historical data with columns: ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'].
    """
    print(len(existing_data))
    if len(existing_data) == 0:
        # If no existing data, fetch the last 2500 bars
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 2500)
        df = pd.DataFrame(rates)
    else:
        # If existing data is provided, fetch only the latest bar
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
        new_data = pd.DataFrame(rates)

        # Concatenate the new data to the existing data
        df = pd.concat([existing_data, new_data])

    # Convert data to DataFrame
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    return df

def calculate_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate common trade patterns such as double tops & bottoms, pennants, wedges, and bull and bear flags.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing price and indicator information.

    Returns:
    - pd.DataFrame: The DataFrame with added columns for detected patterns.
    """
    # Detect Double Tops & Bottoms
    df['double_top'] = np.where((df['high'].shift(1) > df['high']) & (df['high'].shift(1) > df['high'].shift(2)), 'Double Top', 'None')
    df['double_bottom'] = np.where((df['low'].shift(1) < df['low']) & (df['low'].shift(1) < df['low'].shift(2)), 'Double Bottom', 'None')

    # Detect Bull and Bear Flags
    df['bull_flag'] = np.where((df['close'] > abstract.BBANDS(df['close'], timeperiod=5, nbdevup=2.0, nbdevdn=2.0)[0]) & (df['close'].shift(1) < abstract.BBANDS(df['close'].shift(1), timeperiod=5, nbdevup=2.0, nbdevdn=2.0)[0]), 'Bull Flag', 'None')
    df['bear_flag'] = np.where((df['close'] < abstract.BBANDS(df['close'], timeperiod=5, nbdevup=2.0, nbdevdn=2.0)[2]) & (df['close'].shift(1) > abstract.BBANDS(df['close'].shift(1), timeperiod=5, nbdevup=2.0, nbdevdn=2.0)[2]), 'Bear Flag', 'None')

    # Assign patterns based on conditions
    df['pattern'] = 'None'
    conditions = [
        (df['double_top'] != 'None'),
        (df['double_bottom'] != 'None'),
        (df['bull_flag'] != 'None'),
        (df['bear_flag'] != 'None')
    ]

    choices = ['Double Top', 'Double Bottom', 'Bull Flag', 'Bear Flag']
    df['pattern'] = np.select(conditions, choices, default='None')

    return df

def calculate_indicators_and_detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trade signals based on divergences, patterns, and trend direction.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing indicators, patterns, and trend information.

    Returns:
    - pd.DataFrame: The DataFrame with added columns for trade signals.
    """
    # Calculate indicators and detect patterns
    # Add your indicator calculation and pattern detection logic here

    # Example: Calculate RSI
    df['rsi'] = abstract.RSI(df['close'], timeperiod=14)
    # print(df['close'].values)

    # Example: Detect RSI divergence
    df['rsi_divergence'] = (df['rsi'] > 70) & (df['close'] < df['close'].shift()) 

    # Example: Detect TREND signal
    df['trend_signal'] = 'None'
    df['short_ma'] = df['close'].rolling(window=50).mean()
    df['long_ma'] = df['close'].rolling(window=200).mean()

    df.loc[df['short_ma'] > df['long_ma'], 'trend_signal'] = 'Uptrend'
    df.loc[df['short_ma'] < df['long_ma'], 'trend_signal'] = 'Downtrend'
    # Add your MACD divergence detection logic here

    # Example: Detect patterns
    df['pattern'] = 'None'
    # Add your pattern detection logic here
    df = calculate_patterns(df)

    return df

def generate_trade_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trade signals based on divergences, patterns, and trend direction.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing indicators, patterns, and trend information.

    Returns:
    - pd.DataFrame: The DataFrame with added columns for trade signals.
    """
    # Determine trade signals based on divergences, patterns, and trend direction
    df['signal'] = 'None'
    df['divergence_signal'] = np.where((df['rsi_divergence'] == True) & (df['pattern'] != 'None'), 'Both',
                                       np.where(df['rsi_divergence'] == True, 'RSI', 'Pattern'))
    df['strongest_divergence_signal'] = df[['divergence_signal']].max(axis=1)

    # Additional conditions for trade signals
    conditions = [
        (df['strongest_divergence_signal'] != 'None'),
        # Placeholder for 'resistance' calculation - replace this with your actual logic
        (df['close'] > df['close'].rolling(window=10).max()),
        (df['close'] < df['close'].rolling(window=10).min()),
        # Use the calculated 'trend_signal' column for trend condition
        (df['trend_signal'] == 'Uptrend'),
        (df['trend_signal'] == 'Downtrend'),
        # Additional condition to check if the pattern is valid
        (df['pattern'] != 'None'),
    ]

    choices = ['Divergence', 'Resistance', 'Support', 'Uptrend', 'Downtrend', 'Pattern']

    # Ensure that the lengths of conditions and choices are the same
    if len(conditions) == len(choices):
        df['support_resistance_signal'] = np.select(conditions, choices, default='None')
    else:
        # Handle the case where lengths do not match (print an error message for debugging)
        print("Error: Lengths of conditions and choices do not match.")
        df['support_resistance_signal'] = 'None'

    # Iterate over the data points
    for i in range(1, len(df)):
        strongest_divergence_signal = df['strongest_divergence_signal'].iloc[i]
        support_resistance_signal = df['support_resistance_signal'].iloc[i]
        trend_signal = df['trend_signal'].iloc[i]

        if strongest_divergence_signal != 'None':
            df.loc[df.index[i], 'signal'] = strongest_divergence_signal
        elif support_resistance_signal != 'None':
            df.loc[df.index[i], 'signal'] = support_resistance_signal
        elif trend_signal != 'None':
            df.loc[df.index[i], 'signal'] = trend_signal

    return df

def execute_trade(signal_priority, df, symbol, lot_size, stop_loss, take_profit):
    """
    Execute a trade based on the provided signal and trading parameters.

    Parameters:
    - signal_priority (int): The priority assigned to the trade signal.
    - df (pd.DataFrame): The DataFrame containing trade-related information.
    - symbol (str): The financial instrument symbol (e.g., 'EURUSD').
    - lot_size (float): The size of the trading position.
    - stop_loss (float): The stop-loss level.
    - take_profit (float): The take-profit level.
    """
    for index, row in df.iterrows():

        # Additional conditions for Buy trade
        if (
            (signal_priority == 3 and row['rsi_divergence'] and row['rsi_value'] < 30 and row['trend_signal'] == 'Downtrend') or
            (signal_priority == 2 and row['pattern'] == 'Double Bottom' and row['trend_signal'] == 'Downtrend') or
            (signal_priority == 1 and 40 <= row['rsi_value'] <= 60 and row['pattern'] == 'Bull Flag' and row['trend_signal'] == 'Uptrend') or
            (signal_priority == 0 and row['rsi_value'] < 30 and row['rsi_divergence'] and row['pattern'] == 'Bull')
        ):
            # Place a buy trade
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': symbol,
                'volume': lot_size,
                'type': mt5.ORDER_TYPE_BUY,
                'price': mt5.symbol_info_tick(symbol).ask,
                'sl': mt5.symbol_info_tick(symbol).ask - stop_loss * mt5.symbol_info(symbol).point,
                'tp': mt5.symbol_info_tick(symbol).ask + take_profit * mt5.symbol_info(symbol).point,
                'deviation': 0,
                'magic': 123456,
                'comment': "Buy trade",
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }
        elif (
            (signal_priority == 3 and row['rsi_divergence'] and row['rsi_value'] > 70 and row['trend_signal'] == 'Uptrend') or
            (signal_priority == 2 and row['pattern'] == 'Double Top' and row['trend_signal'] == 'Uptrend') or
            (signal_priority == 1 and 40 <= row['rsi_value'] <= 60 and row['pattern'] == 'Bear Flag' and row['trend_signal'] == 'Downtrend') or
            (signal_priority == 0 and row['rsi_value'] > 70 and row['rsi_divergence'] and row['pattern'] == 'Bear')
        ):
            # Place a sell trade
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': symbol,
                'volume': lot_size,
                'type': mt5.ORDER_TYPE_SELL,
                'price': mt5.symbol_info_tick(symbol).bid,
                'sl': mt5.symbol_info_tick(symbol).bid + stop_loss * mt5.symbol_info(symbol).point,
                'tp': mt5.symbol_info_tick(symbol).bid - take_profit * mt5.symbol_info(symbol).point,
                'deviation': 0,
                'magic': 123456,
                'comment': "Sell trade",
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }

        try:
            if signal_priority != 0:
                
                result = mt5.order_send(request)
                print(result)
                outcome = 'Win' if result.retcode == mt5.TRADE_RETCODE_DONE else 'Loss'

                # Example trade outcome information
                trade_outcome = {
                    'pattern': row['pattern'],
                    'divergence_strength': row['strongest_divergence_signal'],
                    'time': pd.Timestamp.now(),
                    'trend_direction': row['trend_signal'],
                    'indicator_used': row['strongest_divergence_signal'],
                    'outcome': outcome
                }
                # Update TensorFlow neural network model with trade outcome
                neural_network_model.update_neural_network_model(trade_outcome, 'tradedata.csv')

                # Example print statements for debugging
                print(
                    f"Executed trade with signal priority: {signal_priority}, position size: {lot_size}")
                print(f"Trade outcome: {trade_outcome}")

                # Additional logic for trade management, monitoring, etc.
        except Exception as e:
            print(f"Error executing trade: {str(e)}")
