import torch
import torch.optim as optim
import torch.nn as nn
from itertools import product
from sklearn.metrics import mean_absolute_error, mean_squared_error
from model import Encoder

def train_model(X_train, y_train, learning_rate, num_layers, input_dim, d_model, nhead, dim_feedforward, batch_size, num_epochs):
    model = Encoder(input_dim, d_model, nhead, num_layers, dim_feedforward)
    criterion = nn.L1Loss()
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
