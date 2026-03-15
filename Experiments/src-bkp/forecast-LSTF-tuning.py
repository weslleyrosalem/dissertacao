import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2

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

# Create sequences of input data and target values for train set
X_train, y_train = [], []
for i in range(n_steps, len(train_scaled)):
    X_train.append(train_scaled[i-n_steps:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the input data to be 3D
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_features))

# Define the Transformer model
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = tf.keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = tf.keras.layers.Dropout(dropout)(x)
    res = x + inputs

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

inputs = tf.keras.layers.Input(shape=(n_steps, n_features))
x = transformer_encoder(inputs, head_size=64, num_heads=1, ff_dim=128, dropout=0.1)
outputs = tf.keras.layers.Dense(1, kernel_regularizer=l2(0.01))(x[:, -1, :])

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='rmsprop', loss='mae') # Change optimizer to RMSprop

# Learning Rate Schedule
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-3 * 10**(epoch / 20))
# Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

callbacks = [lr_schedule, early_stopping]

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=callbacks, verbose=0)

# ...rest of your code for predictions, evaluation and plotting...
num_iterations = 10

# List to store metrics
rmse_list = []
mae_list = []
test_preds = []
train_preds = []


# Loop to train the model multiple times
for i in range(num_iterations):
    # Train the model
    model.fit(X_train, y_train, epochs=10, verbose=0)

    # Create sequences of input data and target values for test set
    X_test, y_test = [], []
    for j in range(n_steps, len(test_scaled)):
        X_test.append(test_scaled[j-n_steps:j, 0])
        y_test.append(test_scaled[j, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Reshape the input data to be 3D
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], n_features))

    # Make predictions on the test set
    test_pred = model.predict(X_test)
    test_pred = test_pred[:, -1]  # Take the last prediction of each sequence
    test_pred = scaler.inverse_transform(test_pred.reshape(-1, 1))
    test_preds.append(test_pred)

    # Make predictions on the train set
    train_pred = model.predict(X_train)
    train_pred = train_pred[:, -1]  # Take the last prediction of each sequence
    train_pred = scaler.inverse_transform(train_pred.reshape(-1, 1))
    train_preds.append(train_pred)

    # Evaluate the model on the test set and store metrics
    rmse = np.sqrt(np.mean((test_pred - test.values[n_steps:]) ** 2))
    rmse_list.append(rmse)
    mae = mean_absolute_error(test.values[n_steps:], test_pred)
    mae_list.append(mae)

    # print metrics for each iteration if you want
    print(f"Iteration {i + 1} - RMSE: {rmse}, MAE: {mae}")

# Calculate the mean metrics
mean_rmse = np.mean(rmse_list)
mean_mae = np.mean(mae_list)

print("Mean RMSE:", mean_rmse)
print("Mean MAE:", mean_mae)

# Calculate the mean predicted train values
mean_train_preds = np.mean(train_preds, axis=0)

# Plot the actual vs predicted values for the train and test sets
plt.figure(figsize=(10, 6))
plt.plot(train.index[n_steps:], train.values[n_steps:], label="Actual Train")
plt.plot(train.index[n_steps:], mean_train_preds, label="Predicted Train")
plt.plot(test.index[n_steps:], test.values[n_steps:], label="Actual Test")
plt.plot(test.index[n_steps:], np.mean(test_preds, axis=0), label="Predicted Test")
plt.legend()
plt.show()