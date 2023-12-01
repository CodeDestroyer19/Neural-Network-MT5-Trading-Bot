"""
Main script for running the trading bot with a web interface.

This script initializes the MetaTrader 5 connection, runs the trading bot, and integrates with a web interface for user credentials.

Author: Mike Kiwalabye
"""

import time
from flask import Flask, render_template, request, redirect, json, Response
from src.connectors import mt5_connector
from src.models import neural_network_model
from src.strategies.trading_strategy import get_historical_data, calculate_indicators_and_detect_patterns, generate_trade_signals, execute_trade
from src.utils.visualization import plot_trade_signals
import threading
import pandas as pd

app = Flask(__name__)

# Define input shape for the neural network
input_shape = (11,)  # Adjust the input shape based on your features and data

# Create the neural network model
neural_network_model = neural_network_model.create_neural_network_model(input_shape)

# Global state to track whether MT5 is initialized
mt5_initialized = False
latest_trade_signals = []


# Web Interface Routes

@app.route('/')
def index():
    """Render the main page with the login form."""
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    """
    Handle the login form submission.

    If the credentials are valid, start the trading bot with the provided credentials.

    Returns:
    - str: HTML response.
    """
    global mt5_initialized
    if request.method == 'POST':
        credentials = {
            'username': request.form['username'],
            'password': request.form['password'],
            'server': request.form['server'],
            'path': request.form['path']
        }
        if mt5_connector.connect_to_mt5(credentials):
            # Set MT5 initialization state to True
            mt5_initialized = True
            # Redirect to the main dashboard or another page
            return redirect('/dashboard')
        else:
            return render_template('index.html', error='Invalid credentials. Please try again.')

@app.route('/dashboard')
def dashboard():
    # Replace these with the actual MetaTrader data retrieval logic
    mt5_data = mt5_connector.get_account_info()  # Replace with the actual method to get account info
    user_data = {'username': mt5_data.name, 'account_balance': mt5_data.balance, 'currency': mt5_data.currency}
    username = user_data.get('username', 'N/A')
    account_balance = user_data.get('account_balance', 'N/A')
    account_currency = user_data.get('currency', 'N/A')

    return render_template('dashboard.html', username=username, account_balance=account_balance, account_currency=account_currency)

# Flask app routes

@app.route('/start_ml_bot', methods=['POST'])
def start_ml_bot():
    """
    Handle the request to start the ML bot.
    """
    # Start the ML bot
    threading.Thread(target=run_trading_bot_web_interface).start()

    # Return an empty response
    return Response(status=200)


@app.route("/stop_ml_bot", methods=['GET'])
def stop_ml_bot():
    mt5_connector.stop_mt5_ml_bot()
    redirect('/dashboard')

def map_signal_priority(signal_priority):
    # Define a mapping for string values to integers
    signal_mapping = {
        'Both': 1,
        'Pattern': 2,
        'RSI': 3
        # Add more mappings as needed
    }

    # Use the mapping, default to 0 if not found
    return signal_mapping.get(signal_priority, 0)
# Main Trading Bot Logic 

def run_trading_bot_web_interface():
    """
    Run the trading bot using MetaTrader 5 credentials from the web interface.
    """
    global latest_trade_signals
    historical_data_df = pd.DataFrame()
    
    while True:
        try:
            symbol = 'EURUSD'
            lot_size = 0.01
            stop_loss = 100
            take_profit = 200

            # Get the latest historical data
            historical_data_df = get_historical_data(symbol, historical_data_df)

            # Calculate indicators and detect patterns for the latest data
            df = calculate_indicators_and_detect_patterns(historical_data_df)

            # Generate trade signals for the latest data
            df = generate_trade_signals(df)
            df.to_csv('your_file.csv', sep='\t', index=False)

            # Inside the run_trading_bot_web_interface function
            latest_trade_signals = df.replace({pd.NA: 'null'}).to_json(orient='records')


            # Execute trades
            for i in range(len(df)):
                
                signal_priority = df['signal'].iloc[i]  # Replace with your actual value
                mapped_priority = map_signal_priority(signal_priority)

                if mapped_priority != 0:
                    execute_trade(mapped_priority, df, symbol, lot_size, stop_loss, take_profit)

        except Exception as e:
            print(f"Error running trading bot: {str(e)}")

        # Wait for the next iteration
        time.sleep(60)  # Adjust the time interval as needed

@app.route('/get_latest_trade_signals', methods=['GET'])
def get_latest_trade_signals():
    global latest_trade_signals
    return json.dumps(latest_trade_signals)

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
