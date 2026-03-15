import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

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
