import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt

def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def create_model(learning_rate=0.001, num_layers=1, n_steps=3, d_model=64, num_heads=4, loss='mse', dropout_rate=0.2):
    inputs = tf.keras.Input(shape=(n_steps, d_model))
    x = inputs
    for _ in range(num_layers):
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x += inputs
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss)
    return model

try:
    metric_df = pd.read_pickle("../data/ts.pkl")
    ts = metric_df["value"].astype(float).resample("30min").mean()
    train = ts[:"2021-02-07"]
    test = ts["2021-02-08":]
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    test_scaled = scaler.transform(test.values.reshape(-1, 1))

    n_steps = 3
    n_features = 1
    d_model = 64

    X_train, y_train = [], []
    for i in range(n_steps, len(train_scaled)):
        X_train.append(train_scaled[i-n_steps:i, 0])
        y_train.append(train_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_features))
    X_train += positional_encoding(n_steps, d_model)

    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'num_layers': [1, 2, 3],
        'n_steps': [3, 5, 7],
        'd_model': [64, 128],
        'num_heads': [2, 4],
        'loss': ['mse', 'mae'],
        'dropout_rate': [0.2, 0.5],
        'batch_size': [16, 32, 64],
        'epochs': [30, 50, 100]
    }
    
    model = KerasRegressor(build_fn=create_model, verbose=0)
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=50, n_jobs=-1, cv=3, verbose=1)
    random_search_result = random_search.fit(X_train, y_train)

    best_model = create_model(**random_search_result.best_params_)
    best_model.fit(X_train, y_train, epochs=random_search_result.best_params_['epochs'], batch_size=random_search_result.best_params_['batch_size'], verbose=1)

except Exception as e:
    print("An error occurred:", e)

# Prediction and evaluation
try:
    X_test, y_test = [], []
    for i in range(n_steps, len(test_scaled)):
        X_test.append(test_scaled[i-n_steps:i, 0])
        y_test.append(test_scaled[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], n_features))
    X_test += positional_encoding(n_steps, d_model)

    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)

    train_pred = train_pred[:, -1, 0]
    test_pred = test_pred[:, -1, 0]

    train_pred = scaler.inverse_transform(train_pred.reshape(-1, 1))
    test_pred = scaler.inverse_transform(test_pred.reshape(-1, 1))

    rmse = np.sqrt(np.mean((test_pred.flatten() - test.values[n_steps:]) ** 2))
    print(f"RMSE: {rmse}")

    mae = mean_absolute_error(test.values[n_steps:], test_pred.flatten())
    print(f"MAE: {mae}")

    plt.figure(figsize=(10, 6))
    plt.plot(train.index[n_steps:], train.values[n_steps:], label="Train Actual")
    plt.plot(train.index[n_steps:], train_pred.flatten(), label="Train Predicted")
    plt.plot(test.index[n_steps:], test.values[n_steps:], label="Test Actual")
    plt.plot(test.index[n_steps:], test_pred.flatten(), label="Test Predicted")
    plt.legend()
    plt.title("Actual vs. Predicted Values")
    plt.show()

except NameError as e:
    print("It seems the 'best_model' was not defined, likely due to an exception in an earlier stage:", e)
