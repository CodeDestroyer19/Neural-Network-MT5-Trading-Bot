from keras.optimizers import Adam
from keras.layers import Dense, Dropout
from keras.models import Sequential
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import talib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def connect_to_mt5_container():
    server = "localhost"  # Change to the appropriate IP or hostname if necessary
    port = 15555  # Change to the appropriate port if necessary
    login = 123456  # Change to your MetaTrader login number if necessary
    password = "your_password"  # Change to your MetaTrader password if necessary

    # Initialize MetaTrader 5
    mt5.initialize()

    # Connect to MetaTrader 5 server
    connected = mt5.login(login, password, server=server, port=port)
    if not connected:
        print("Failed to connect to MetaTrader 5")
        return False

    print(f"Connected to MetaTrader 5: {mt5.terminal_info()}")
    return True


def start_mt5_bot():
    # Define the symbols and timeframes
    symbol = 'EURUSD'
    timeframe = mt5.TIMEFRAME_H1  # H1 timeframe (1 hour)

    # Set up initial variables
    lot_size = 0.01
    stop_loss = 100
    take_profit = 150

    # Define TensorFlow neural network model
    def create_neural_network_model(input_shape):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        return model

    # Define input shape for the neural network
    # Adjust the input shape based on your features and data
    input_shape = (10,)

    # Create the neural network model
    neural_network_model = create_neural_network_model(input_shape)

    # Compile the model
    neural_network_model.compile(optimizer=Adam(
        learning_rate=0.001), loss='binary_crossentropy')

    def get_historical_data():
        # Retrieve historical data
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1000)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df

    def calculate_indicators_and_detect_patterns(df):
        # Calculate RSI
        rsi_period = 14
        df['rsi'] = talib.RSI(df['close'], rsi_period)

        # Calculate MACD
        macd_fast_period = 12
        macd_slow_period = 26
        macd_signal_period = 9
        macd, macd_signal, _ = talib.MACD(df['close'], fastperiod=macd_fast_period,
                                          slowperiod=macd_slow_period, signalperiod=macd_signal_period)
        df['macd'] = macd
        df['macd_signal'] = macd_signal

        # Detect divergence based on RSI and MACD
        df['rsi_divergence'] = np.where(
            df['rsi'].diff().shift(-1) * df['macd'].diff().shift(-1) < 0, True, False)
        df['macd_divergence'] = np.where(
            df['macd'].diff().shift(-1) * df['rsi'].diff().shift(-1) < 0, True, False)

        # Detect support and resistance levels
        window = 10
        df['support'] = df['low'].rolling(window).min()
        df['resistance'] = df['high'].rolling(window).max()

        # Determine trend direction
        df['trend_200'] = df['close'].rolling(window=200).mean()
        df['trend_50'] = df['close'].rolling(window=50).mean()

        # Detect double tops and bottoms
        df['pattern'] = 'None'
        df['top_pattern'] = np.where(
            (df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high']) &
            (df['high'].shift(2) > df['high']) & (
                df['high'].shift(-2) > df['high']), 'Double Top', 'None'
        )
        df.loc[df['top_pattern'] != 'None', 'pattern'] = df['top_pattern']
        df['bottom_pattern'] = np.where(
            (df['low'].shift(1) > df['low']) & (df['low'].shift(-1) > df['low']) &
            (df['low'].shift(2) < df['low']) & (
                df['low'].shift(-2) < df['low']), 'Double Bottom', 'None'
        )
        df.loc[df['bottom_pattern'] != 'None',
               'pattern'] = df['bottom_pattern']

        return df

    def generate_signals(df):
        # Determine trade signals based on divergences, patterns, and trend direction
        df['signal'] = 'None'
        df['divergence_signal'] = np.where((df['rsi_divergence'] == True) & (df['pattern'] != 'None'), 'Both',
                                           np.where(df['rsi_divergence'] == True, 'RSI', 'Pattern'))
        df['strongest_divergence_signal'] = df[[
            'divergence_signal', 'macd_divergence']].max(axis=1)
        df['support_resistance_signal'] = np.where(df['close'] > df['resistance'], 'Resistance',
                                                   np.where(df['close'] < df['support'], 'Support', 'None'))
        df['trend_signal'] = np.where(df['close'] > df['trend_200'], 'Uptrend',
                                      np.where(df['close'] < df['trend_200'], 'Downtrend', 'None'))
        for i in range(1, len(df)):
            prev_divergence_signal = df['divergence_signal'].iloc[i - 1]
            curr_divergence_signal = df['divergence_signal'].iloc[i]
            strongest_divergence_signal = df['strongest_divergence_signal'].iloc[i]
            support_resistance_signal = df['support_resistance_signal'].iloc[i]
            trend_signal = df['trend_signal'].iloc[i]

            if strongest_divergence_signal != 'None':
                df['signal'].iloc[i] = strongest_divergence_signal
            elif prev_divergence_signal == curr_divergence_signal and curr_divergence_signal != 'None':
                df['signal'].iloc[i] = curr_divergence_signal
            else:
                df['signal'].iloc[i] = support_resistance_signal

            if trend_signal != 'None' and df['signal'].iloc[i] != 'None':
                df['signal'].iloc[i] = trend_signal

        return df

    def execute_trade(signal, df):
        # Implement risk management and trade execution logic based on the signals generated
        # Update TensorFlow neural network model with trade outcome (loss or win)

        # Calculate risk and position size based on lot size, stop loss, and take profit
        risk = lot_size * stop_loss
        strongest_divergence_signal = df['strongest_divergence_signal'].iloc[-1]
        if strongest_divergence_signal == 'RSI':
            risk *= 1.2  # Increase risk by 20% if RSI divergence is the strongest
        elif strongest_divergence_signal == 'Pattern':
            risk *= 1.5  # Increase risk by 50% if pattern divergence is the strongest

        position_size = risk / (take_profit - stop_loss)

        try:
            if signal == 'Buy':
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
                    'type_filling': mt5.ORDER_FILLING_RETURN,
                }
                result = mt5.order_send(request)
                outcome = 'Win' if result.retcode == mt5.TRADE_RETCODE_DONE else 'Loss'

            elif signal == 'Sell':
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
                    'type_filling': mt5.ORDER_FILLING_RETURN,
                }
                result = mt5.order_send(request)
                outcome = 'Win' if result.retcode == mt5.TRADE_RETCODE_DONE else 'Loss'

            # Example trade outcome information
            trade_outcome = {
                'pattern': df['pattern'].iloc[-1],
                'divergence_strength': strongest_divergence_signal,
                'time': df.index[-1],
                'trend_direction': df['trend_signal'].iloc[-1],
                'indicator_used': strongest_divergence_signal,
                'outcome': outcome
            }

            # Update TensorFlow neural network model with trade outcome
            update_neural_network_model(trade_outcome)

            # Example print statements for debugging
            print(
                f"Executed {signal} trade with position size: {position_size}")
            print(f"Trade outcome: {trade_outcome}")

            # Additional logic for trade management, monitoring, etc.
        except Exception as e:
            print(f"Error executing trade: {str(e)}")

    def update_neural_network_model(trade_outcome):
        # Implement code to update the neural network model based on trade outcome
        pattern = trade_outcome['pattern']
        divergence_strength = trade_outcome['divergence_strength']
        time = trade_outcome['time']
        trend_direction = trade_outcome['trend_direction']
        indicator_used = trade_outcome['indicator_used']
        outcome = trade_outcome['outcome']

        # Example update code: Append trade outcome information to a dataset for future training
        trade_data = pd.DataFrame({
            'pattern': [pattern],
            'divergence_strength': [divergence_strength],
            'time': [time],
            'trend_direction': [trend_direction],
            'indicator_used': [indicator_used],
            'outcome': [outcome]
        })
        # Append the trade data to the dataset for future training
        dataset = pd.read_csv('trade_dataset.csv')  # Load existing dataset
        updated_dataset = pd.concat([dataset, trade_data], ignore_index=True)
        # Save updated dataset
        updated_dataset.to_csv('trade_dataset.csv', index=False)

        # Example retraining code: Retrain the neural network model with the updated dataset
        # Preprocess data as per your requirements
        X_train, y_train = preprocess_data(updated_dataset)
        # Example retraining step
        neural_network_model.fit(X_train, y_train, epochs=10, batch_size=32)

        # Save the updated model weights
        neural_network_model.save_weights('weights/model_weights.h5')

    def preprocess_data(dataset):
        # Define the numerical features (if any)
        numerical_features = []  # Update with the actual numerical feature column names

        # Define the input features
        input_features = dataset[[
            'pattern', 'divergence_strength', 'trend_direction', 'indicator_used']]

        # Convert categorical features to one-hot encoding
        input_features = pd.get_dummies(input_features)

        # Normalize numerical features (if any)
        if numerical_features:
            scaler = MinMaxScaler()
            input_features[numerical_features] = scaler.fit_transform(
                input_features[numerical_features])

        # Extract target labels from the dataset
        target_labels = dataset['outcome']

        # Convert target labels to numerical representation (0s and 1s)
        target_labels = target_labels.map({'Loss': 0, 'Win': 1})

        # Return the preprocessed input features and target labels
        return input_features, target_labels

    def visualize_data(df):
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['close'], label='Close')
        # Add visualizations for other indicators, levels, and patterns
        plt.scatter(df[df['pattern'] == 'Double Top'].index, df[df['pattern'] == 'Double Top']['high'],
                    color='red', marker='v', label='Double Top')
        plt.scatter(df[df['pattern'] == 'Double Bottom'].index, df[df['pattern'] == 'Double Bottom']['low'],
                    color='green', marker='^', label='Double Bottom')
        plt.legend()
        plt.show()

    def run_trading_bot():
        # Connect to MetaTrader 5 container
        connected = connect_to_mt5_container()
        if not connected:
            return

        while True:
            try:
                # Get historical data
                df = get_historical_data()

                # Calculate indicators and detect patterns
                df = calculate_indicators_and_detect_patterns(df)

                # Generate trade signals
                df = generate_signals(df)

                # Execute trades
                for i in range(1, len(df)):
                    signal = df['signal'].iloc[i]
                    if signal != 'None':
                        execute_trade(signal, df)

                # Visualize data
                visualize_data(df)

            except Exception as e:
                print(f"Error running trading bot: {str(e)}")

            # Wait for the next iteration
            time.sleep(60)  # Adjust the time interval as needed

    # Run the trading bot
    run_trading_bot()

    # Disconnect from MetaTrader 5
    mt5.shutdown()


# Start the MetaTrader 5 bot
start_mt5_bot()
