import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import product

# Function to run the training loop for a given set of hyperparameters
def train_model(learning_rate, num_layers, X_train, y_train, X_test, y_test, input_dim, d_model, nhead, dim_feedforward, batch_size, num_epochs):
    # Initialize model, loss, and optimizer with the given hyperparameters
    model = Encoder(input_dim, d_model, nhead, num_layers, dim_feedforward)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)

    # Rescale the predicted values
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))

    return mae, rmse

# Load the dataset
file_path = '../data/ts.pkl'
df = pd.read_pickle(file_path)
df = df['value'].astype(float).resample('1T').mean().fillna(method='ffill')
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.values.reshape(-1, 1))

# Create sequences
SEQ_LENGTH = 60
X, y = [], []
for i in range(len(df_scaled) - SEQ_LENGTH):
    X.append(df_scaled[i:i + SEQ_LENGTH])
    y.append(df_scaled[i + SEQ_LENGTH])

X = np.array(X)
y = np.array(y)

# Expand feature dimension to match 'd_model'
d_model = 64
X = np.repeat(X, d_model, axis=2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the Transformer Encoder model
class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward):
        super(Encoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear_out = nn.Linear(d_model, 1)

    def forward(self, src):
        src = src.permute(1, 0, 2)
        output = self.transformer_encoder(src)
        output = self.linear_out(output[-1])
        return output

# Hyperparameters to tune
learning_rates = [0.001, 0.01]
num_layers_list = [2, 3]

# Other fixed hyperparameters
input_dim = d_model
nhead = 4
dim_feedforward = 256
batch_size = 64
num_epochs = 10

# Initialize the best MAE and RMSE to infinity and the best hyperparameters to None
best_mae = float('inf')
best_rmse = float('inf')
best_hyperparams = None

# Grid search
for learning_rate, num_layers in product(learning_rates, num_layers_list):
    mae, rmse = train_model(learning_rate, num_layers, X_train, y_train, X_test, y_test, input_dim, d_model, nhead, dim_feedforward, batch_size, num_epochs)
    print(f'Learning Rate: {learning_rate}, Num Layers: {num_layers}, MAE: {mae}, RMSE: {rmse}')

    # Update the best MAE and RMSE and best hyperparameters if needed
    if mae < best_mae:
        best_mae = mae
        best_rmse = rmse
        best_hyperparams = {'learning_rate': learning_rate, 'num_layers': num_layers}

print(f'Best Hyperparameters: {best_hyperparams}, Best MAE: {best_mae}, Best RMSE: {best_rmse}')