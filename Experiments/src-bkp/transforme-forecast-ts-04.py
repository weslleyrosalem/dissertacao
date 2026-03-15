import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import product

# Function Definitions

def load_and_preprocess_data():
    metric_df = pd.read_pickle("../data/ts.pkl")
    ts = metric_df["value"].astype(float).resample("30min").mean()
    ts.fillna(method='ffill', inplace=True)
    return ts

def create_sequences(ts_scaled, SEQ_LENGTH, d_model):
    X, y = [], []
    for i in range(len(ts_scaled) - SEQ_LENGTH):
        X.append(ts_scaled[i:i + SEQ_LENGTH])
        y.append(ts_scaled[i + SEQ_LENGTH])

    X = np.array(X)
    y = np.array(y)
    X = np.repeat(X, d_model, axis=2)
    return X, y


# Expand feature dimension to match 'd_model'
d_model = 64
X = np.repeat(X, d_model, axis=2)

# Train-test split based on the date

# Ensure the time series index is unique
if not ts.index.is_unique:
    print("The time series index must be unique. Please resolve this before proceeding.")
    exit()

# Get the location of the specified date
train_size = ts.index.get_loc("2021-02-07")

# Convert to integer explicitly, if needed
if isinstance(train_size, slice):
    print("Multiple indices found for the date. Using the start of the slice.")
    train_size = train_size.start

# Convert train_size to integer
train_size = int(train_size)

# Now, we can safely slice the data
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Further implementation continues here...


# Define the Model
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

# Training Loop
def train_model(X_train, y_train, learning_rate, num_layers, input_dim, d_model, nhead, dim_feedforward, batch_size, num_epochs):
    model = Encoder(input_dim, d_model, nhead, num_layers, dim_feedforward)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    return model

# Main Execution Block
if __name__ == "__main__":
    # Constants and Hyperparameters
    SEQ_LENGTH = 60  
    d_model = 64
    input_dim = d_model
    nhead = 4
    dim_feedforward = 256
    batch_size = 64
    num_epochs = 10
    learning_rates = [0.001, 0.01]
    num_layers_list = [2, 3]
    
    # Load and preprocess data
    ts = load_and_preprocess_data()
    scaler = StandardScaler()
    ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1))
    X, y = create_sequences(ts_scaled, SEQ_LENGTH, d_model)
    
    # Train-Test Split
    train_size = ts.index.get_loc("2021-02-07")
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



