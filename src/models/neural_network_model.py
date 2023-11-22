# neural_network_model.py
"""
Module for defining and managing the TensorFlow Neural Network model.

This module provides functions to create, compile, and update a simple feedforward neural network
model for use in a trading bot.

Author: Mike Kiwalabye
"""

import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def create_neural_network_model(input_shape: tuple) -> Sequential:
    """
    Create a simple feedforward neural network model.

    Parameters:
    - input_shape (tuple): The shape of the input data.

    Returns:
    - Sequential: The Keras Sequential model representing the neural network.
    """
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model


def compile_neural_network_model(model: Sequential, learning_rate: float = 0.001) -> None:
    """
    Compile the neural network model.

    Parameters:
    - model (Sequential): The Keras Sequential model representing the neural network.
    - learning_rate (float): The learning rate for the Adam optimizer.
    """
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])


def preprocess_data(dataset: pd.DataFrame) -> tuple:
    """
    Preprocess the dataset for training the neural network model.

    Parameters:
    - dataset (pd.DataFrame): The dataset containing trade outcome information.

    Returns:
    - tuple: A tuple containing preprocessed input features and target labels.
    """
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


def update_neural_network_model(trade_outcome: dict, dataset_path: str) -> None:
    """
    Update the neural network model based on trade outcome.

    Parameters:
    - trade_outcome (dict): Trade outcome information.
    - dataset_path (str): The path to the dataset file for updating and saving.
    """
    # Load existing model or create a new one if it doesn't exist
    try:
        model = load_model('model_weights.h5')
    except (OSError, ValueError):
        # If loading fails, create a new model
        input_shape = (4,)  # Replace with the actual input shape
        model = create_neural_network_model(input_shape)
        compile_neural_network_model(model, learning_rate=0.001)

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

    try:
        # Load existing dataset if it exists
        dataset = pd.read_csv(dataset_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # Create an empty dataset if the file doesn't exist or is empty
        dataset = pd.DataFrame()

    # Concatenate the trade data to the dataset for future training
    updated_dataset = pd.concat([dataset, trade_data], ignore_index=True)

    # Save updated dataset
    updated_dataset.to_csv(dataset_path, index=False)

    # Example retraining code: Retrain the neural network model with the updated dataset
    # Preprocess data as per your requirements
    X_train, y_train = preprocess_data(updated_dataset)  # Implement the preprocess_data function

    # Convert labels to NumPy array and ensure the correct data type
    y_train = np.array(y_train).astype(float)  # Convert to float

    # Ensure labels have the correct shape
    y_train = y_train.reshape(-1)

    # Example retraining step
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Save the updated model weights
    model.save_weights('model_weights.h5')
