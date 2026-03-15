import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Function for positional encoding
def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# Load the time series data
metric_df = pd.read_pickle("../data/ts.pkl")
ts = metric_df["value"].astype(float).resample("30min").mean()
train = ts[:"2021-02-07"]
test = ts["2021-02-08":]

# Scale the data using a MinMaxScaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
test_scaled = scaler.transform(test.values.reshape(-1, 1))

# Hyperparameters
n_steps = 3
n_features = 1
d_model = 64
num_heads = 4
num_layers = 20

# Create sequences for train and test sets
X_train, y_train = [], []
for i in range(n_steps, len(train_scaled)):
    X_train.append(train_scaled[i-n_steps:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape and add positional encoding
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_features))
X_train += positional_encoding(n_steps, d_model)

# Transformer Encoder Layer
def encoder_layer(units, d_model, num_heads):
    inputs = tf.keras.Input(shape=(None, d_model))
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x += inputs

    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Stacking multiple encoder layers for deep architecture
inputs = tf.keras.Input(shape=(n_steps, d_model))
x = inputs
for _ in range(num_layers):
    x = encoder_layer(512, d_model, num_heads)(x)

outputs = tf.keras.layers.Dense(1, activation='linear')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile and train the model
model.compile(optimizer='adam', loss='mae')
model.fit(X_train, y_train, epochs=500, verbose=0)

# Prepare test data
X_test, y_test = [], []
for i in range(n_steps, len(test_scaled)):
    X_test.append(test_scaled[i-n_steps:i, 0])
    y_test.append(test_scaled[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], n_features))
X_test += positional_encoding(n_steps, d_model)

# Make predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Reshape the predictions to be 2D
train_pred = train_pred[:, -1, 0]
test_pred = test_pred[:, -1, 0]

# Inverse scaling
train_pred = scaler.inverse_transform(train_pred.reshape(-1, 1))
test_pred = scaler.inverse_transform(test_pred.reshape(-1, 1))

# Metrics
rmse = np.sqrt(np.mean((test_pred.flatten() - test.values[n_steps:]) ** 2))
print(f"RMSE: {rmse}")

mae = mean_absolute_error(test.values[n_steps:], test_pred.flatten())
print(f"MAE: {mae}")


# Plotting
plt.figure(figsize=(10, 6))

# Plot training data
plt.plot(train.index[n_steps:], train.values[n_steps:], label="Train Actual")
plt.plot(train.index[n_steps:], train_pred.flatten(), label="Train Predicted")

# Plot testing data
plt.plot(test.index[n_steps:], test.values[n_steps:], label="Test Actual")
plt.plot(test.index[n_steps:], test_pred.flatten(), label="Test Predicted")

# Additional plot settings
plt.legend()
plt.title("Actual vs. Predicted Values")
plt.xlabel("Date")
plt.ylabel("Value")
plt.show()