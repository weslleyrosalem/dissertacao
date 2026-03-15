"""
Transformer Encoder – Time-Series Forecasting of Prometheus Memory Metrics
==========================================================================
Corrections applied (v2):
  1. Input projection (nn.Linear) replaces naive np.repeat.
  3. Hyperparameter selection uses the VALIDATION set; the TEST set is
     evaluated only once, with the best configuration, to avoid data leakage.
"""

import torch
import numpy as np
from itertools import product
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_preparation import load_and_preprocess_data, create_sequences, split_and_scale
from train import train_model

MB = 1_048_576

# ─── Hyperparameter space ────────────────────────────────────────────
RESAMPLE = "15min"
SEQ_LENGTH = 48       # 12 hours of context at 15-min resolution
d_model = 64
nhead = 4
dim_feedforward = 256
batch_size = 64
num_epochs = 30

learning_rates = [0.0005, 0.001, 0.005]
num_layers_list = [2, 3, 5]


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.maximum(denominator, 1e-8)
    return 100.0 * np.mean(diff)


# ─── Data pipeline ───────────────────────────────────────────────────
print("=" * 60)
print("Loading and preprocessing data …")
ts = load_and_preprocess_data(resample_interval=RESAMPLE)
print(f"  Total samples : {len(ts)}")
print(f"  Date range    : {ts.index[0]}  →  {ts.index[-1]}")

train_scaled, val_scaled, test_scaled, scaler = split_and_scale(ts)
print(f"  Train : {len(train_scaled)}  |  Val : {len(val_scaled)}  |  Test : {len(test_scaled)}")

X_train, y_train = create_sequences(train_scaled, SEQ_LENGTH)
X_val, y_val = create_sequences(val_scaled, SEQ_LENGTH)
X_test, y_test = create_sequences(test_scaled, SEQ_LENGTH)

print(f"  X_train shape : {X_train.shape}  (no np.repeat – raw 1-dim input)")
print(f"  X_val   shape : {X_val.shape}")
print(f"  X_test  shape : {X_test.shape}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# ─── Hyperparameter search (VALIDATION set only) ─────────────────────
print("\n" + "=" * 60)
print("Hyperparameter search  (evaluated on VALIDATION set)")
print("=" * 60)

best_val_loss = float('inf')
best_hyperparams = None
best_model = None
results_log = []

for lr, n_layers in product(learning_rates, num_layers_list):
    tag = f"lr={lr}, layers={n_layers}"
    print(f"\n▸ {tag}")

    model, val_loss = train_model(
        X_train_t, y_train_t, X_val_t, y_val_t,
        learning_rate=lr,
        num_layers=n_layers,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_size=batch_size,
        num_epochs=num_epochs,
        patience=10,
    )

    model.eval()
    with torch.no_grad():
        y_vp = model(X_val_t).numpy()

    y_vp_mb = scaler.inverse_transform(y_vp) / MB
    y_v_mb = scaler.inverse_transform(y_val) / MB

    val_mae = mean_absolute_error(y_v_mb, y_vp_mb)
    val_rmse = np.sqrt(mean_squared_error(y_v_mb, y_vp_mb))
    val_smape = smape(y_v_mb, y_vp_mb)

    results_log.append({
        "lr": lr, "layers": n_layers,
        "val_loss": val_loss, "val_mae": val_mae,
        "val_rmse": val_rmse, "val_smape": val_smape,
    })
    print(f"  Val Loss : {val_loss:.6f}  |  MAE : {val_mae:.4f} MB  |  RMSE : {val_rmse:.4f} MB  |  SMAPE : {val_smape:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_hyperparams = {"learning_rate": lr, "num_layers": n_layers}
        best_model = model
        print("  ★  new best configuration")

# ─── Summary of all configurations ───────────────────────────────────
print("\n" + "=" * 60)
print("Search results (sorted by val_loss)")
print("-" * 60)
for r in sorted(results_log, key=lambda x: x["val_loss"]):
    mark = " ◀ BEST" if r["lr"] == best_hyperparams["learning_rate"] and r["layers"] == best_hyperparams["num_layers"] else ""
    print(f"  lr={r['lr']:<8}  layers={r['layers']}  │  val_loss={r['val_loss']:.6f}  MAE={r['val_mae']:.4f}  RMSE={r['val_rmse']:.4f}  SMAPE={r['val_smape']:.2f}%{mark}")

# ─── Final evaluation on TEST set (single pass, no leakage) ──────────
print("\n" + "=" * 60)
print("FINAL evaluation on TEST set  (best hyperparameters)")
print(f"  Config: {best_hyperparams}")
print("=" * 60)

best_model.eval()
with torch.no_grad():
    y_test_pred = best_model(X_test_t).numpy()

y_test_pred_mb = scaler.inverse_transform(y_test_pred) / MB
y_test_mb = scaler.inverse_transform(y_test) / MB

test_mae = mean_absolute_error(y_test_mb, y_test_pred_mb)
test_rmse = np.sqrt(mean_squared_error(y_test_mb, y_test_pred_mb))
test_mape = mean_absolute_percentage_error(y_test_mb, y_test_pred_mb) * 100
test_smape = smape(y_test_mb, y_test_pred_mb)

print(f"  MAE   : {test_mae:.4f} MB")
print(f"  RMSE  : {test_rmse:.4f} MB")
print(f"  MAPE  : {test_mape:.2f} %")
print(f"  SMAPE : {test_smape:.2f} %")
print("=" * 60)
