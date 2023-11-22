# Trading Bot

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A trading bot built using Python and TensorFlow to automate trading strategies.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The trading bot is designed to automate trading strategies using historical data, technical indicators, and machine learning. It connects to the MetaTrader 5 platform, retrieves historical data, calculates indicators, generates trade signals, executes trades, and updates a neural network model based on trade outcomes.

## Features

- Retrieval of historical data from MetaTrader 5
- Calculation of technical indicators (RSI, MACD, etc.)
- Detection of double tops and bottoms patterns
- Generation of trade signals based on indicators, patterns, and trend direction
- Execution of trades with risk management
- Update of a TensorFlow neural network model based on trade outcomes
- Visualization of data and trade signals

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/your-username/trading-bot.git
   ```

1. Install Docker and Docker Compose on your system.
1. Build the Docker image and start the container:

   ```shell
   cd trading-bot
   docker-compose up -d --build
   ```

## Usage

1. Ensure that the Docker container is running.
1. Access the running container:

   ```shell
   docker exec -it trading-bot_app_1 bash
   ```

1. Inside the container, run the trading bot:

   ```shell
   python main.py
   ```

1. The trading bot will start executing the trading strategies based on the predefined logic.
1. Monitor the bot's output and visualizations.

1. To stop the bot, use `Ctrl + C` in the terminal.

## Configuration

The trading bot can be customized and configured by modifying the following files:

- `main.py`: Contains the main logic for running the trading bot.
- `config.py`: Defines the configuration parameters such as symbol, timeframe, lot size, stop loss, take profit, etc.
- `indicators.py`: Defines additional technical indicators and patterns to be used.
- `preprocess.py`: Handles data preprocessing and feature engineering.
- `model.py`: Defines the structure and training of the neural network model.
- `docker-compose.yml`: Configures the Docker container for running the trading bot.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please feel free to submit a pull request or create an issue in the repository.

## License

This project is licensed under the [MIT Licence](https://opensource.org/license/mit/).
