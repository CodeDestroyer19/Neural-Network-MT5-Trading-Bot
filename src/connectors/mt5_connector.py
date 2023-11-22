# mt5_connector.py

"""
Module for connecting to MetaTrader 5 (MT5) using the MetaTrader5 Python package.

This module provides functions for initializing and connecting to MT5, along with error handling.

Author: Mike Kiwalabye
"""

import MetaTrader5 as mt5


def start_mt5(username: str, password: str, server: str, path: str) -> bool:
    """
    Initialize and start a connection to MetaTrader 5.

    Parameters:
    - username (str): The MT5 account username.
    - password (str): The MT5 account password.
    - server (str): The MT5 trading server.
    - path (str): The file path to the MetaTrader 5 executable.

    Returns:
    - bool: True if successfully initialized, False otherwise.
    """
    # Ensure that all variables are the correct type
    uname = int(username)  # Username must be an int
    pword = str(password)  # Password must be a string
    trading_server = str(server)  # Server must be a string
    filepath = str(path)  # Filepath must be a string

    # Connect to MetaTrader 5
    if mt5.initialize(login=uname, password=pword, server=trading_server, path=filepath):
        # Login to MT5
        if mt5.login(login=uname, password=pword, server=trading_server):
            return True
        else:
            print("Login Fail")
            quit()
            return PermissionError
    else:
        print("MT5 Initialization Failed")
        quit()
        return ConnectionAbortedError

def connect_to_mt5(credentials: dict):
    """
    Connect to MetaTrader 5.

    Parameters:
    - credentials (dict): Dictionary containing 'username', 'password', 'server', and 'path'.
    """
    # Start the MetaTrader 5 instance
    if start_mt5(credentials['username'], credentials['password'], credentials['server'], credentials['path']):
        print("Connected to MetaTrader 5")
        return True
    else:
        print("Failed to connect to MetaTrader 5")
        return False

def get_account_info():
    """
    Get account information from MetaTrader 5.

    Returns:
    - dict: Dictionary containing account information (e.g., username, account balance).
    """

    # Fetch account information
    account_info = mt5.account_info()

    return account_info