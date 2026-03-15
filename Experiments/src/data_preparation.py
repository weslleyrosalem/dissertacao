import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(filepath="../data/ts.pkl", resample_interval="30min"):
    metric_df = pd.read_pickle(filepath)
    ts = metric_df["value"].astype(float)
    ts = ts.rolling(window=3, min_periods=1).mean()
    ts = ts.resample(resample_interval).mean()
    ts = ts.ffill().dropna()
    return ts


def create_sequences(data, seq_length):
    """Build sliding-window sequences.  Returns X with shape (N, seq_length, 1)."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def split_and_scale(ts, train_ratio=0.6, val_ratio=0.2):
    """60 / 20 / 20 chronological split with StandardScaler fitted on train only."""
    n = len(ts)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    train = ts.iloc[:train_end]
    val = ts.iloc[train_end:val_end]
    test = ts.iloc[val_end:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    val_scaled = scaler.transform(val.values.reshape(-1, 1))
    test_scaled = scaler.transform(test.values.reshape(-1, 1))

    return train_scaled, val_scaled, test_scaled, scaler
