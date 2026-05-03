from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wifi_uav.config import ProjectConfig


def _user_features(group: pd.DataFrame, cfg: ProjectConfig) -> tuple[np.ndarray, np.ndarray]:
    group = group.sort_values("step")
    wifi_cols = [f"wifi_rssi_{idx}" for idx in range(cfg.n_aps)]
    cell_cols = [f"cell_rssi_{idx}" for idx in range(cfg.n_base_stations)]
    csi_cols = ["csi_amp_mean", "csi_amp_std", "csi_amp_max", "csi_phase_mean"]

    wifi = group[wifi_cols].to_numpy(dtype=np.float32)
    delta = np.vstack([np.zeros((1, cfg.n_aps), dtype=np.float32), np.diff(wifi, axis=0)])
    rolling_var = (
        group[wifi_cols]
        .rolling(window=3, min_periods=1)
        .var()
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    cell = group[cell_cols].to_numpy(dtype=np.float32)
    csi = group[csi_cols].to_numpy(dtype=np.float32)
    features = np.concatenate([wifi, delta, rolling_var, cell, csi], axis=1).astype(np.float32)
    targets = group[["x", "y"]].to_numpy(dtype=np.float32)
    return features, targets


def build_sequences(df: pd.DataFrame, cfg: ProjectConfig) -> tuple[np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for (_, _), group in df.groupby(["episode", "user_id"], sort=True):
        features, targets = _user_features(group, cfg)
        max_start = len(group) - cfg.seq_len - cfg.pred_horizon + 1
        for start in range(max(0, max_start)):
            end = start + cfg.seq_len
            target_idx = end + cfg.pred_horizon - 1
            windows.append(features[start:end])
            labels.append(targets[target_idx])

    if not windows:
        raise ValueError("No sequences were created; increase steps_per_episode or reduce seq_len/pred_horizon")

    return np.stack(windows).astype(np.float32), np.stack(labels).astype(np.float32)


def build_feature_splits(
    cfg: ProjectConfig,
    csv_path: str | Path,
    output_dir: str | Path,
) -> dict[str, np.ndarray]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    x, y = build_sequences(df, cfg)

    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(x))
    x = x[order]
    y = y[order] / cfg.area_size

    n_train = int(0.70 * len(x))
    n_val = int(0.15 * len(x))
    if n_train == 0 or n_val == 0 or len(x) - n_train - n_val == 0:
        raise ValueError("Not enough sequences for non-empty train/val/test splits")

    train_x = x[:n_train]
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x = ((x - mean) / std).astype(np.float32)
    y = y.astype(np.float32)

    splits = {
        "X_train": x[:n_train],
        "y_train": y[:n_train],
        "X_val": x[n_train : n_train + n_val],
        "y_val": y[n_train : n_train + n_val],
        "X_test": x[n_train + n_val :],
        "y_test": y[n_train + n_val :],
    }

    for name, arr in splits.items():
        np.save(output_dir / f"{name}.npy", arr.astype(np.float32))
    np.savez(
        output_dir / "normalization.npz",
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        area_size=np.float32(cfg.area_size),
    )
    return {name: arr.astype(np.float32) for name, arr in splits.items()}
