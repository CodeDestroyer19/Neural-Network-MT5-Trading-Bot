# data_processing.py
"""
Module for processing and preprocessing data for the trading bot.

This module provides functions for loading, cleaning, and preprocessing historical price data.

Author: Mike Kiwalabye
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load historical price data from a CSV file.

    Parameters:
    - file_path (str): The path to the CSV file containing historical price data.

    Returns:
    - pd.DataFrame: The DataFrame containing historical price data.
    """
    # Load data from CSV file
    df = pd.read_csv(file_path)

    # Ensure the 'time' column is in datetime format
    df['time'] = pd.to_datetime(df['time'])

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the historical price data by handling missing values and removing duplicates.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing historical price data.

    Returns:
    - pd.DataFrame: The cleaned DataFrame.
    """
    # Handle missing values (if any)
    df.dropna(inplace=True)

    # Remove duplicate rows (if any)
    df.drop_duplicates(inplace=True)

    return df

def preprocess_data(df: pd.DataFrame, numerical_features: list = []) -> pd.DataFrame:
    """
    Preprocess the historical price data by normalizing numerical features and encoding categorical features.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing historical price data.
    - numerical_features (list): A list of column names corresponding to numerical features.

    Returns:
    - pd.DataFrame: The preprocessed DataFrame.
    """
    # Convert categorical features to one-hot encoding
    df = pd.get_dummies(df)

    # Normalize numerical features (if any)
    if numerical_features:
        scaler = MinMaxScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df