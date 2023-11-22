"""
Main script for running the trading bot with a web interface.

This script initializes the MetaTrader 5 connection, runs the trading bot, and integrates with a web interface for user credentials.

Author: Mike Kiwalabye
"""

import time
from flask import Flask, render_template, request, redirect, globals
from src.connectors import mt5_connector
from src.models import neural_network_model
from src.strategies.trading_strategy import get_historical_data, calculate_indicators_and_detect_patterns, generate_trade_signals, execute_trade
from src.utils.visualization import plot_trade_signals

app = Flask(__name__)

# Define input shape for the neural network
input_shape = (10,)  # Adjust the input shape based on your features and data

# Create the neural network model
neural_network_model = neural_network_model.create_neural_network_model(input_shape)

# Global state to track whether MT5 is initialized
globals.mt5_initialized = False

# Web Interface Routes

@app.route('/')
def index():
    """Render the main page with login form."""
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    """
    Handle the login form submission.

    If the credentials are valid, start the trading bot with the provided credentials.

    Returns:
    - str: HTML response.
    """
    if request.method == 'POST':
        credentials = {
            'username': request.form['username'],
            'password': request.form['password'],
            'server': request.form['server'],
            'path': request.form['path']
        }
        if mt5_connector.connect_to_mt5(credentials):
            # Set MT5 initialization state to True
            globals.mt5_initialized = True
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
    run_trading_bot_web_interface()


# Main Trading Bot Logic

def run_trading_bot_web_interface():
    """
    Run the trading bot using MetaTrader 5 credentials from the web interface.
    """

    while True:
        try:
            symbol = 'EURUSD'
            lot_size = 0.01
            stop_loss = 100
            take_profit = 150

            # Get the latest historical data
            latest_data = get_historical_data(symbol).iloc[-1:]

            # Calculate indicators and detect patterns for the latest data
            df = calculate_indicators_and_detect_patterns(latest_data)

            # Generate trade signals for the latest data
            df = generate_trade_signals(df)
            print(df)
            # Execute trades
            for i in range(len(latest_data)):
                signal = df['signal'].iloc[i]
                if signal != 'None':
                    print(df['signal'].array)
                    execute_trade(signal, df, symbol, lot_size, stop_loss, take_profit)

            # Visualize data
            # plot_trade_signals(df)

        except Exception as e:
            print(f"Error running trading bot: {str(e)}")

        # Wait for the next iteration
        time.sleep(60)  # Adjust the time interval as needed

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
