import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

# Step 1: Load Bitcoin Data (Using Yahoo Finance API)
ticker = 'BTC-USD'
data = yf.download(ticker, start='2015-01-01', end='2024-01-01')

# Step 2: Preprocess the data
# Use the 'Close' price for prediction
data = data[['Close']]

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Step 3: Prepare the data for LSTM (Time-Series data)
look_back = 60  # Use 60 previous days to predict the next day

def create_dataset(data, look_back=look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])  # Previous 'look_back' days
        y.append(data[i, 0])  # Current day
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)

# Reshape X to be suitable for LSTM input (samples, time steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Step 4: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 5: Build LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))  # Output layer for predicting the Bitcoin price

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 6: Train the Model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Step 7: Make Predictions
predicted_prices = model.predict(X_test)

# Inverse scaling to get the actual values
predicted_prices = scaler.inverse_transform(predicted_prices)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 8: Plot the Results
plt.figure(figsize=(10,6))
plt.plot(y_test_actual, color='blue', label='Actual Bitcoin Price')
plt.plot(predicted_prices, color='red', label='Predicted Bitcoin Price')
plt.title('Bitcoin Price Prediction (Test Data)')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price (USD)')
plt.legend()
plt.show()
