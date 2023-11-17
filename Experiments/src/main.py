import torch
import numpy as np
from itertools import product
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from data_preparation import load_and_preprocess_data, create_sequences
from train import train_model
from model import Encoder

# Constants and Hyperparameters
SEQ_LENGTH = 60  
d_model = 64
input_dim = d_model
nhead = 4
dim_feedforward = 256
batch_size = 64
num_epochs = 20
learning_rates = [0.001, 0.01]
num_layers_list = [2, 3, 5, 10]

# Load and preprocess data
ts = load_and_preprocess_data()
scaler = StandardScaler()
ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1))
X, y = create_sequences(ts_scaled, SEQ_LENGTH, d_model)

# Train-Test Split
train_size_loc = ts.index.get_loc("2021-02-07")

if isinstance(train_size_loc, slice):
    print("Multiple indices found for the date. Using the start of the slice.")
    train_size = train_size_loc.start
elif isinstance(train_size_loc, int):
    train_size = train_size_loc
else:
    print("Unexpected type for index location. Exiting.")
    exit(1)

train_size = int(train_size)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Train and Evaluate Model
best_mae = float('inf')
best_rmse = float('inf')
best_hyperparams = None

for learning_rate, num_layers in product(learning_rates, num_layers_list):
    model = train_model(X_train, y_train, learning_rate, num_layers, input_dim, d_model, nhead, dim_feedforward, batch_size, num_epochs)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy()
    
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test)
    
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
    
    if mae < best_mae:
        best_mae = mae
        best_rmse = rmse
        best_hyperparams = {'learning_rate': learning_rate, 'num_layers': num_layers}

print(f'Best Hyperparameters: {best_hyperparams}, Best MAE: {best_mae}, Best RMSE: {best_rmse}')
