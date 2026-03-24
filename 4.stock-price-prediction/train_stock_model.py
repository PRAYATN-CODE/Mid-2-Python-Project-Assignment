import pandas as pd
import numpy as np
import yfinance as yf
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

# 1. Fetch Global Time-Series Data (Bitcoin as default for global appeal)
ticker = 'BTC-USD'
print(f"Downloading historical data for {ticker}...")
data = yf.download(ticker, start='2020-01-01', end='2024-01-01')
df = data[['Close']].dropna()

# 2. Time-series Data Preprocessing
print("Scaling and preparing sequences...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Create sequences (Look back 60 days to predict the next day)
sequence_length = 60

def create_sequences(dataset, seq_length):
    X, y = [], []
    for i in range(len(dataset) - seq_length):
        X.append(dataset[i:(i + seq_length), 0])
        y.append(dataset[i + seq_length, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, sequence_length)

# Train-Test Split (Chronological, NOT randomized!)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for Deep Learning [samples, time steps, features]
X_train_dl = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_dl = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 3 & 4. Implementation and Comparison of Models
print("\n--- Training Traditional Model (Random Forest) ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train) # RF takes 2D data
rf_predictions = rf_model.predict(X_test)
print(f"Random Forest MSE: {mean_squared_error(y_test, rf_predictions):.5f}")

print("\n--- Training Deep Learning Model 1: LSTM ---")
lstm_model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train_dl.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_dl, y_train, batch_size=32, epochs=10, validation_split=0.1, verbose=1)
lstm_predictions = lstm_model.predict(X_test_dl)
lstm_mse = mean_squared_error(y_test, lstm_predictions)
print(f"LSTM MSE: {lstm_mse:.5f}")

print("\n--- Training Deep Learning Model 2: GRU ---")
gru_model = Sequential([
    GRU(units=50, return_sequences=True, input_shape=(X_train_dl.shape[1], 1)),
    Dropout(0.2),
    GRU(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])
gru_model.compile(optimizer='adam', loss='mean_squared_error')
gru_model.fit(X_train_dl, y_train, batch_size=32, epochs=10, validation_split=0.1, verbose=1)
gru_predictions = gru_model.predict(X_test_dl)
gru_mse = mean_squared_error(y_test, gru_predictions)
print(f"GRU MSE: {gru_mse:.5f}")

# Select the best deep learning model based on MSE
best_model = lstm_model if lstm_mse < gru_mse else gru_model
best_name = "LSTM" if lstm_mse < gru_mse else "GRU"
print(f"\nBest DL Model selected: {best_name}")

# Save the model, scaler, and configuration
best_model.save('best_stock_model.h5')
with open('stock_scaler.pkl', 'wb') as f:
    pickle.dump({'scaler': scaler, 'seq_length': sequence_length}, f)

print("Pipeline complete! Artifacts saved.")