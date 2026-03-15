"""
Transformer Encoder – Full Experiment Suite
============================================
Replicates the dissertation experiments at 5min, 15min, and 30min resolutions
using the corrected architecture (linear projection + no data leakage).

Generates plots matching the dissertation style and saves them with "new-" prefix.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from model import Encoder

# ─── Constants ────────────────────────────────────────────────────────
MB = 1_048_576
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FILE_PATH = os.path.join(DATA_DIR, "ts.pkl")

EXPERIMENTS = [
    {"resample": "5min",  "seq_length": 144, "label": "5min"},
    {"resample": "15min", "seq_length": 48,  "label": "15min"},
    {"resample": "30min", "seq_length": 24,  "label": "30min"},
]

D_MODEL = 64
NHEAD = 4
DIM_FF = 256
BATCH_SIZE = 64
NUM_EPOCHS = 50
PATIENCE = 10
LEARNING_RATES = [0.0005, 0.001, 0.005]
NUM_LAYERS_LIST = [2, 3, 5]


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return 100.0 * np.mean(np.abs(y_true - y_pred) / np.maximum(denom, 1e-8))


# ─── Data helpers ─────────────────────────────────────────────────────
def load_data(resample_interval):
    df = pd.read_pickle(FILE_PATH)
    ts = df["value"].astype(float)
    ts = ts.rolling(window=3, min_periods=1).mean()
    ts = ts.resample(resample_interval).mean()
    ts = ts.ffill().dropna()
    return ts


def split_and_scale(ts):
    n = len(ts)
    t1 = int(0.6 * n)
    t2 = t1 + int(0.2 * n)
    train, val, test = ts.iloc[:t1], ts.iloc[t1:t2], ts.iloc[t2:]

    scaler = StandardScaler()
    train_s = scaler.fit_transform(train.values.reshape(-1, 1))
    val_s = scaler.transform(val.values.reshape(-1, 1))
    test_s = scaler.transform(test.values.reshape(-1, 1))
    return train_s, val_s, test_s, scaler, train.index, val.index, test.index


def make_sequences(data, dates, seq_len):
    X, y, d = [], [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
        d.append(dates[i + seq_len])
    return np.array(X), np.array(y), np.array(d)


# ─── Training ─────────────────────────────────────────────────────────
def train_model(X_tr, y_tr, X_vl, y_vl, lr, n_layers):
    model = Encoder(D_MODEL, NHEAD, n_layers, DIM_FF)
    crit = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)

    best_vl, best_st, cnt = float("inf"), None, 0

    for _ in range(NUM_EPOCHS):
        model.train()
        idx = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), BATCH_SIZE):
            b = idx[i:i + BATCH_SIZE]
            out = model(X_tr[b])
            loss = crit(out, y_tr[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vl = crit(model(X_vl), y_vl).item()
        if vl < best_vl:
            best_vl = vl
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
            cnt = 0
        else:
            cnt += 1
            if cnt >= PATIENCE:
                break

    if best_st:
        model.load_state_dict(best_st)
    return model, best_vl


# ─── Plotting (matching dissertation style) ───────────────────────────
def generate_plot(model, scaler, X_tr, y_tr, d_tr, X_te, y_te, d_te,
                  resample_label, metrics, save_path):
    model.eval()
    with torch.no_grad():
        p_tr = model(X_tr).numpy()
        p_te = model(X_te).numpy()

    y_tr_mb = scaler.inverse_transform(y_tr.numpy()) / MB
    p_tr_mb = scaler.inverse_transform(p_tr) / MB
    y_te_mb = scaler.inverse_transform(y_te.numpy()) / MB
    p_te_mb = scaler.inverse_transform(p_te) / MB

    train_df = pd.DataFrame({
        "date": d_tr, "actual": y_tr_mb.flatten(), "predicted": p_tr_mb.flatten()
    }).sort_values("date")
    test_df = pd.DataFrame({
        "date": d_te, "actual": y_te_mb.flatten(), "predicted": p_te_mb.flatten()
    }).sort_values("date")

    plt.style.use("default")
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=False)

    axs[0].plot(train_df["date"], train_df["actual"],
                label="Real", color="blue", linewidth=1.5)
    axs[0].plot(train_df["date"], train_df["predicted"],
                label="Predito", color="red", alpha=0.7, linewidth=1.5)
    axs[0].set_title("Conjunto de Treinamento (60%)", fontsize=12, pad=10)
    axs[0].set_ylabel("Consumo de Memória (MB)", fontsize=10)
    axs[0].legend(loc="upper left", fontsize=10)
    axs[0].grid(True, linestyle="--", alpha=0.7)

    axs[1].plot(test_df["date"], test_df["actual"],
                label="Real", color="blue", linewidth=1.5)
    axs[1].plot(test_df["date"], test_df["predicted"],
                label="Predito", color="red", alpha=0.7, linewidth=1.5)
    axs[1].set_title("Conjunto de Teste (20%)", fontsize=12, pad=10)
    axs[1].set_xlabel("Data", fontsize=10)
    axs[1].set_ylabel("Consumo de Memória (MB)", fontsize=10)
    axs[1].legend(loc="upper left", fontsize=10)
    axs[1].grid(True, linestyle="--", alpha=0.7)

    for ax in axs:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)

    metrics_text = (
        f"MAE: {metrics['mae']:.2f} MB  |  RMSE: {metrics['rmse']:.2f} MB  "
        f"|  MAPE: {metrics['mape']:.2f}%  |  SMAPE: {metrics['smape']:.2f}%"
    )
    fig.text(0.5, 0.01, metrics_text, ha="center", fontsize=10, style="italic",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    plt.suptitle(
        f"Predições do Transformer Corrigido - Prometheus (MB, Resample {resample_label})",
        fontsize=14, y=0.98,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved -> {save_path}")


# ─── Main loop ────────────────────────────────────────────────────────
def run_experiment(exp):
    resample = exp["resample"]
    seq_len = exp["seq_length"]
    label = exp["label"]

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: Transformer — Resample {label}  (SEQ_LENGTH={seq_len})")
    print(f"{'='*70}")

    ts = load_data(resample)
    print(f"  Samples: {len(ts)}  |  {ts.index[0]}  ->  {ts.index[-1]}")

    tr_s, vl_s, te_s, scaler, tr_idx, vl_idx, te_idx = split_and_scale(ts)
    X_tr, y_tr, d_tr = make_sequences(tr_s, tr_idx, seq_len)
    X_vl, y_vl, d_vl = make_sequences(vl_s, vl_idx, seq_len)
    X_te, y_te, d_te = make_sequences(te_s, te_idx, seq_len)

    print(f"  Train seqs: {X_tr.shape[0]}  |  Val seqs: {X_vl.shape[0]}  |  Test seqs: {X_te.shape[0]}")

    if X_vl.shape[0] < 2 or X_te.shape[0] < 2:
        print("  WARNING: Not enough sequences - skipping this resolution.")
        return None

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_vl_t = torch.tensor(X_vl, dtype=torch.float32)
    y_vl_t = torch.tensor(y_vl, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.float32)

    n_configs = len(LEARNING_RATES) * len(NUM_LAYERS_LIST)
    print(f"\n  Grid search ({n_configs} configs) on VALIDATION set ...")
    best_vl_loss, best_hp, best_model = float("inf"), None, None
    results_log = []

    for lr, nl in product(LEARNING_RATES, NUM_LAYERS_LIST):
        model, vl_loss = train_model(X_tr_t, y_tr_t, X_vl_t, y_vl_t, lr, nl)
        model.eval()
        with torch.no_grad():
            vp = model(X_vl_t).numpy()
        vp_mb = scaler.inverse_transform(vp) / MB
        v_mb = scaler.inverse_transform(y_vl) / MB
        v_mae = mean_absolute_error(v_mb, vp_mb)
        v_rmse = np.sqrt(mean_squared_error(v_mb, vp_mb))

        results_log.append({"lr": lr, "layers": nl, "val_loss": vl_loss,
                            "val_mae": v_mae, "val_rmse": v_rmse})
        mark = ""
        if vl_loss < best_vl_loss:
            best_vl_loss = vl_loss
            best_hp = {"lr": lr, "layers": nl}
            best_model = model
            mark = " *"
        print(f"    lr={lr:<8} layers={nl}  val_loss={vl_loss:.6f}  MAE={v_mae:.2f}  RMSE={v_rmse:.2f}{mark}")

    print(f"\n  Best config: {best_hp}")

    best_model.eval()
    with torch.no_grad():
        tp = best_model(X_te_t).numpy()
    tp_mb = scaler.inverse_transform(tp) / MB
    te_mb = scaler.inverse_transform(y_te) / MB

    metrics = {
        "mae": mean_absolute_error(te_mb, tp_mb),
        "rmse": np.sqrt(mean_squared_error(te_mb, tp_mb)),
        "mape": mean_absolute_percentage_error(te_mb, tp_mb) * 100,
        "smape": smape(te_mb, tp_mb),
    }

    print(f"\n  TEST RESULTS:")
    print(f"    MAE   : {metrics['mae']:.4f} MB")
    print(f"    RMSE  : {metrics['rmse']:.4f} MB")
    print(f"    MAPE  : {metrics['mape']:.2f} %")
    print(f"    SMAPE : {metrics['smape']:.2f} %")

    plot_name = f"new-prometheus_transformer_mrfo_{label}.png"
    save_path = os.path.join(DATA_DIR, plot_name)
    generate_plot(
        best_model, scaler,
        X_tr_t, y_tr_t, d_tr,
        X_te_t, y_te_t, d_te,
        label, metrics, save_path,
    )

    return {
        "resample": label, "best_hp": best_hp, **metrics,
        "n_train": X_tr.shape[0], "n_test": X_te.shape[0],
    }


# ─── Run all ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_results = []
    for exp in EXPERIMENTS:
        r = run_experiment(exp)
        if r:
            all_results.append(r)

    print(f"\n\n{'='*80}")
    print("  SUMMARY - All Experiments (Corrected Transformer)")
    print(f"{'='*80}")
    header = f"  {'Resample':<10} {'Config':<22} {'MAE (MB)':<12} {'RMSE (MB)':<12} {'MAPE %':<10} {'SMAPE %':<10}"
    print(header)
    print(f"  {'-'*76}")
    for r in all_results:
        cfg = f"lr={r['best_hp']['lr']}, L={r['best_hp']['layers']}"
        print(f"  {r['resample']:<10} {cfg:<22} {r['mae']:<12.4f} {r['rmse']:<12.4f} {r['mape']:<10.2f} {r['smape']:<10.2f}")
    print(f"{'='*80}")

    file_list = ", ".join(
        "new-prometheus_transformer_mrfo_" + r["resample"] + ".png"
        for r in all_results
    )
    print(f"\n  Images saved in: {DATA_DIR}")
    print(f"  Files: {file_list}")
