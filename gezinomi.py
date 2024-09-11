# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 21:38:21 2024

@author: efeme
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
df = pd.read_excel('miuul_gezinomi.xlsx')  # Use relative path for portability

# Check the general structure of the dataset
df.info()

# Clean the data by dropping rows with null Price
df_cleaned = df.dropna(subset=['Price'])

# Feature engineering: Extracting year, month, and day from dates
df_cleaned['SaleYear'] = df_cleaned['SaleDate'].dt.year
df_cleaned['SaleMonth'] = df_cleaned['SaleDate'].dt.month
df_cleaned['SaleDay'] = df_cleaned['SaleDate'].dt.day

df_cleaned['CheckInYear'] = df_cleaned['CheckInDate'].dt.year
df_cleaned['CheckInMonth'] = df_cleaned['CheckInDate'].dt.month
df_cleaned['CheckInDay'] = df_cleaned['CheckInDate'].dt.day

# Exploratory Data Analysis
# Group by city and season to calculate average price
city_season_price = df_cleaned.groupby(['SaleCityName', 'Seasons'])['Price'].mean().reset_index()

# Group by year and month to calculate average price trends
year_month_price = df_cleaned.groupby(['SaleYear', 'SaleMonth'])['Price'].mean().reset_index()

# Prophet Model Function
def run_prophet_model(data, periods=12):
    """ Trains and forecasts using Prophet model on provided data. """
    model = Prophet()
    model.fit(data)
    
    # Make future dataframe for prediction
    future = model.make_future_dataframe(periods=periods, freq='M')
    
    # Predict
    forecast = model.predict(future)
    
    # Plot the forecast
    model.plot(forecast)
    plt.show()
    
    return forecast

# Prepare data for Prophet (Original)
prophet_df = df_cleaned.reset_index()[['SaleDate', 'Price']]
prophet_df.rename(columns={'SaleDate': 'ds', 'Price': 'y'}, inplace=True)

# Run Prophet on original data
forecast = run_prophet_model(prophet_df)

# Smoothing the Price column using a rolling window (12 months)
df_cleaned['Price_Smoothed'] = df_cleaned['Price'].rolling(window=12).mean()

# Prepare data for Prophet (Smoothed)
prophet_smoothed_df = df_cleaned.reset_index()[['SaleDate', 'Price_Smoothed']]
prophet_smoothed_df.rename(columns={'SaleDate': 'ds', 'Price_Smoothed': 'y'}, inplace=True)

# Run Prophet on smoothed data
forecast_smoothed = run_prophet_model(prophet_smoothed_df.dropna())

# Outlier removal: define limits
lower_limit = 1
upper_limit = df_cleaned['Price'].quantile(0.99)

# Remove outliers
df_no_outliers = df_cleaned[(df_cleaned['Price'] > lower_limit) & (df_cleaned['Price'] < upper_limit)]

# Apply smoothing after outlier removal
df_no_outliers['Price_Smoothed'] = df_no_outliers['Price'].rolling(window=12).mean()

# Prepare data for Prophet (Outliers Removed and Smoothed)
prophet_no_outliers_df = df_no_outliers.reset_index()[['SaleDate', 'Price_Smoothed']]
prophet_no_outliers_df.rename(columns={'SaleDate': 'ds', 'Price_Smoothed': 'y'}, inplace=True)

# Run Prophet on outliers-removed and smoothed data
forecast_no_outliers = run_prophet_model(prophet_no_outliers_df.dropna())

# Calculate performance metrics
def calculate_metrics(actual, predicted):
    """Calculates MAE, MSE, and RMSE between actual and predicted values."""
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# Calculate metrics for all models
actual_1 = df_cleaned['Price'].iloc[:len(forecast)].dropna()
predicted_1 = forecast['yhat'].head(len(actual_1)).dropna()

actual_2 = df_cleaned['Price'].iloc[:len(forecast_smoothed)].dropna()
predicted_2 = forecast_smoothed['yhat'].head(len(actual_2)).dropna()

actual_3 = df_cleaned['Price'].iloc[:len(forecast_no_outliers)].dropna()
predicted_3 = forecast_no_outliers['yhat'].head(len(actual_3)).dropna()

# Print metrics for each model
mae_1, mse_1, rmse_1 = calculate_metrics(actual_1, predicted_1)
print(f"Prophet Model 1 - MAE: {mae_1}, MSE: {mse_1}, RMSE: {rmse_1}")

mae_2, mse_2, rmse_2 = calculate_metrics(actual_2, predicted_2)
print(f"Prophet Model 2 (Smoothed) - MAE: {mae_2}, MSE: {mse_2}, RMSE: {rmse_2}")

mae_3, mse_3, rmse_3 = calculate_metrics(actual_3, predicted_3)
print(f"Prophet Model 3 (Outliers Removed & Smoothed) - MAE: {mae_3}, MSE: {mse_3}, RMSE: {rmse_3}")
