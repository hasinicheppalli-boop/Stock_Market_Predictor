# Stock Price Prediction using LSTM

## Project Overview
Forecasting stock prices has long been a challenging task for researchers and analysts. Investors are highly interested in predicting future stock movements to make informed decisions. This project builds an **LSTM (Long Short-Term Memory) neural network** to predict stock prices based on historical data, providing supportive information about the future direction of the stock market.

## Dataset
- Source: [Kaggle S&P 500 Stocks Dataset](https://www.kaggle.com/datasets).
- Columns used: `date`, `symbol`, `close`.
- In this project, we focus on **Microsoft (MSFT)** because it has complete data.

## Methodology
1. **Data Preprocessing**
   - Load dataset and filter for MSFT.
   - Handle missing values.
   - Scale closing prices between 0 and 1 using MinMaxScaler.

2. **Sequence Creation**
   - Use 60 days of historical data to predict the 61st day.

3. **Model Architecture**
   - LSTM layers with dropout for regularization.
   - Dense output layer for price prediction.

4. **Training**
   - Optimizer: Adam
   - Loss: Mean Squared Error
   - Epochs: 50
   - Batch size: 32
   - Validation split: 10%

5. **Evaluation**
   - Predictions compared against actual test data.
   - RMSE calculated for accuracy.

6. **Visualization**
   - Line chart showing actual vs. predicted prices.
   - Training vs. validation loss curves.

7. **Future Prediction**
   - Predict the next trading day’s closing price.

## Results
- The model provides supportive insights into stock price trends.
- RMSE metric is used to evaluate prediction accuracy.
- Visualization helps compare actual vs. predicted values.
- Example output:
