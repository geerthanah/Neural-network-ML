# Financial Forecasting with Neural Networks in R

This project uses multilayer perceptron (MLP) neural networks to forecast exchange rates (USD/EUR) based on historical data.

## Overview

- Loads and preprocesses exchange rate data.
- Creates lagged features for time series forecasting.
- Normalizes data for neural network training.
- Trains multiple MLP models with different architectures (one and two hidden layers) and activation functions (`logistic`, `tanh`).
- Evaluates model performance using RMSE, MAE, MAPE, and SMAPE.
- Identifies the best performing models based on RMSE.
- Visualizes predicted vs actual exchange rates.

## Key Libraries

- `readxl` for reading Excel files.
- `neuralnet` for building neural networks.
- `Metrics` and `MLmetrics` for performance evaluation.
- `dplyr`, `tidyverse`, `zoo` for data manipulation and lagging.

## Results

- Performance metrics for multiple MLP architectures.
- Total parameter counts for best models.
- Graphical comparison of predicted and actual exchange rates.

## Usage

Replace the data file path with your own dataset and run the script to train and evaluate models.

---


