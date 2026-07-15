import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# 1. LOAD & PREPROCESS THE DATA

print("Loading Kaggle stock dataset...")
df = pd.read_csv("C:/PersonalFiles/Hasini/pythonProject/sp500_stocks/sp500_stocks.csv")
df.columns = df.columns.str.lower()

target_stock = 'MSFT'
df = df[df['symbol'].astype(str).str.upper() == target_stock].copy()
df = df.dropna(subset=['close'])

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

prices = df['close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

train_size = int(len(scaled_prices) * 0.8)
if train_size < 60:
    raise ValueError("Not enough data to create 60-day sequences.")

train_data = scaled_prices[:train_size]
test_data = scaled_prices[train_size - 60:]

print(f"Found {len(df)} clean data rows for {target_stock}. Preparing data windows...")


# 2. CREATE TIME-SERIES SEQUENCES

X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# 3. BUILD AND TRAIN THE LSTM MODEL

print("Training the AI model...")
model = Sequential([
    Input(shape=(X_train.shape[1], 1)),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)


# 4. TEST THE MODEL

print("Making predictions on test data...")
X_test, y_test = [], prices[train_size:]
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# RMSE Evaluation
rmse = np.sqrt(mean_squared_error(y_test[:len(predictions)], predictions))
print("RMSE:", rmse)


# 5. FUTURE PREDICTION

print("\nCalculating the next trading day's prediction...")
last_60_days = scaled_prices[-60:]
X_future = np.array([last_60_days])
X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))

future_prediction = model.predict(X_future)
future_prediction = scaler.inverse_transform(future_prediction)

print("=" * 60)
print(f"AI PREDICTION: The predicted next closing price for {target_stock} is: ${future_prediction[0][0]:.2f}")
print("=" * 60 + "\n")


# 6. VISUALIZE RESULTS

print("Generating project charts...")
plt.figure(figsize=(12, 6))
plt.plot(y_test[:len(predictions)], color='black', label=f"Actual {target_stock} Price")
plt.plot(predictions, color='green', label="AI Predicted Price")
plt.title("Stock Market Predictor AI")
plt.xlabel("Days (Test Period)")
plt.ylabel("Stock Price ($)")
plt.legend()
plt.grid(True)
plt.show()

# Training vs Validation Loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss During Training")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
