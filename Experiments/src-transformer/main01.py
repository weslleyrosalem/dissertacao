import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load the time series data
metric_df = pd.read_pickle("../data/ts.pkl")

# Resample the data to 30-minute intervals
ts = metric_df["value"].astype(float).resample("30min").mean()

# Split the data into train and test sets
train = ts[:"2021-02-07"]
test = ts["2021-02-08":]

# Scale the data using a MinMaxScaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
test_scaled = scaler.transform(test.values.reshape(-1, 1))

# Define the number of time steps and features
n_steps = 3
n_features = 1

# Create sequences of input data and target values for the train set
X_train, y_train = [], []
for i in range(n_steps, len(train_scaled)):
    X_train.append(train_scaled[i-n_steps:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the input data to be 3D
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_features))

# Define the Transformer model architecture
input_layer = Input(shape=(n_steps, n_features))
attention = MultiHeadAttention(num_heads=4, key_dim=2)(input_layer, input_layer)
flatten = Flatten()(attention)
output_layer = Dense(1)(flatten)
model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=5000, verbose=0)

# Create sequences of input data and target values for the test set
X_test, y_test = [], []
for i in range(n_steps, len(test_scaled)):
    X_test.append(test_scaled[i-n_steps:i, 0])
    y_test.append(test_scaled[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)

# Reshape the input data to be 3D
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], n_features))

# Make predictions
train_pred = model.predict(X_train)
train_pred = scaler.inverse_transform(train_pred)
test_pred = model.predict(X_test)
test_pred = scaler.inverse_transform(test_pred)

# Evaluate the model on MAE
mae = mean_absolute_error(test.values[n_steps:], test_pred)
print("MAE:", mae)

# Evaluate the model on RMSE
rmse = np.sqrt(np.mean((test_pred - test.values[n_steps:])**2))
print("RMSE:", rmse)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(train.index[n_steps:], train.values[n_steps:], label="Train Actual")
plt.plot(train.index[n_steps:], train_pred, label="Train Predicted")
plt.plot(test.index[n_steps:], test.values[n_steps:], label="Test Actual")
plt.plot(test.index[n_steps:], test_pred, label="Test Predicted")
plt.legend()
plt.title("Actual vs. Predicted Values")
plt.xlabel("Date")
plt.ylabel("Value")
plt.show()
