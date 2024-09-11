# Robust-Price-Forecasting-using-Prophet-and-Smoothing-Techniques
Project Overview
This project aims to forecast price trends over a period of several years using time series analysis with the Prophet model. The analysis involves handling noisy data, detecting and removing outliers, and applying smoothing techniques to improve the accuracy of predictions.

Goal of the Analysis
The primary goal of this analysis is to develop a robust time series forecasting model that can accurately predict future prices based on historical sales data. The project explores different forecasting techniques to minimize the impact of anomalies and noise, ultimately delivering reliable future price estimates.

Dataset
The dataset consists of sales data that includes various features such as:

SaleDate: The date of sale.
Price: The price of the sale.
SaleCityName: The city where the sale occurred.
Seasons: The season during which the sale was made.
CheckInDate: The date when the customer checked in.
Key Features of the Data:
SaleDate: Time-based data used for forecasting trends.
Price: The target variable for prediction.
Seasons: Categorical data used for seasonality analysis.
City-wise trends: Data was grouped by city to observe price fluctuations in different regions.

Algorithm Used
Prophet Model: An open-source time series forecasting tool developed by Facebook, designed for forecasting univariate time series data.
Three different approaches were used:
Basic Prophet model on the original data.
Prophet with smoothed price data using a moving average.
Prophet with outlier removal and smoothing for improved forecasting.

Results
Among the three models, the third approach—removing outliers and applying smoothing techniques—provided the most reliable predictions. The results show clear trends in the sales price over time, with seasonal variations and notable price peaks.

Model Evaluation:
The models were evaluated based on the following metrics:

Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Model 3 (Outliers Removed and Smoothed Data) was selected as the final model due to its better handling of extreme values and more accurate forecasting.
