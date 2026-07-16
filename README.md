markdown
# 📈 Stock Price Prediction using LSTM

## 📌 Project Overview
Forecasting stock prices has long been a challenging task for researchers and analysts. Investors are highly interested in predicting future stock movements to make informed decisions. This project builds an **LSTM (Long Short-Term Memory) neural network** to predict stock prices based on historical data, providing supportive information about the future direction of the stock market.

## 📊 Dataset
- **Source:** [Kaggle S&P 500 Stocks Dataset](https://www.kaggle.com/datasets).
- **Columns used:** `date`, `symbol`, `close`.
- In this project, we focus on **Microsoft (MSFT)** because it has complete data.

### 🔽 How to Download and Prepare the Dataset
1. Go to Kaggle and search for **S&P 500 Stocks Dataset**.
2. Download the dataset as a CSV file.
3. Save the file in the following path (or update the code to match your own path):
C:/PersonalFiles/Hasini/pythonProject/sp500_stocks/sp500_stocks.csv

Code
4. Ensure the file contains the columns `date`, `symbol`, and `close`.

---

## ⚙️ Methodology
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

---

## 📈 Results
- The model provides supportive insights into stock price trends.
- RMSE metric is used to evaluate prediction accuracy.
- Visualization helps compare actual vs. predicted values.

### 🖥️ Sample Output
AI PREDICTION: The predicted next closing price for MSFT is: $427.35

Code

**Actual vs. Predicted Prices:**

![Actual vs Predicted](attachments/yDguJ2yKZ7v14CxybR2pk.png)

**Training vs Validation Loss:**

![Training vs Validation Loss](attachments/AjhonnzhVsDAPaTxEnNCQ.png)

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-price-prediction.git
Install dependencies:

bash
pip install -r requirements.txt
Run the script:

bash
python main.py
🛠️ Requirements
Python 3.8+

Libraries:

numpy

pandas

matplotlib

scikit-learn

tensorflow / keras

📌 Future Work
Try GRU or Transformer models.

Use multiple features (volume, open, high, low).

Hyperparameter tuning for better accuracy.

Deploy as a web app for interactive predictions.

📝 Conclusion
This project successfully demonstrates a stock price prediction system using LSTM. It satisfies the original problem statement by providing traders, investors, and analysts with supportive information about the future direction of the stock market, backed by historical data, model evaluation, and clear visualizations.
