import socketio
import time

# Connect to the trading bot container
sio = socketio.Client()
# Replace with the appropriate URL and port of your trading bot container
sio.connect('tcp://trading_bot:3000')

# Handle events from the trading bot container


@sio.event
def connect():
    print('Connected to trading bot container')


@sio.event
def disconnect():
    print('Disconnected from trading bot container')


@sio.event
def send_trade_signal(signal):
    print(f'Received trade signal: {signal}')
    # Process the trade signal and execute trades through MetaTrader 5


# Main loop to keep the script running
while True:
    time.sleep(1)
