from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wifi_uav.config import ProjectConfig
from wifi_uav.signal_model import CSI_COLUMNS, place_access_points


def _rssi_weighted_ap_centroid(wifi: np.ndarray, cfg: ProjectConfig) -> np.ndarray:
    ap_xy = place_access_points(cfg)[:, :2].astype(np.float32)
    stable = (wifi - wifi.max(axis=1, keepdims=True)) / 8.0
    weights = np.exp(stable).astype(np.float32)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
    return (weights @ ap_xy).astype(np.float32)


def rf_feature_matrix(wifi: np.ndarray, cell: np.ndarray, csi: np.ndarray, cfg: ProjectConfig) -> np.ndarray:
    """Per-step features for one user's RF time series (rows ordered by time).

    Shared by offline dataset building and the online predictor inside the UAV env so both
    see exactly the same feature definition.
    """
    wifi = np.asarray(wifi, dtype=np.float32)
    delta = np.vstack([np.zeros((1, wifi.shape[1]), dtype=np.float32), np.diff(wifi, axis=0)])
    rolling_var = (
        pd.DataFrame(wifi).rolling(window=3, min_periods=1).var().fillna(0.0).to_numpy(dtype=np.float32)
    )
    rf_centroid = _rssi_weighted_ap_centroid(wifi, cfg)
    return np.concatenate(
        [wifi, delta, rolling_var, np.asarray(cell, dtype=np.float32), np.asarray(csi, dtype=np.float32), rf_centroid],
        axis=1,
    ).astype(np.float32)


def _user_features(group: pd.DataFrame, cfg: ProjectConfig) -> tuple[np.ndarray, np.ndarray]:
    group = group.sort_values("step")
    wifi_cols = [f"wifi_rssi_{idx}" for idx in range(cfg.n_aps)]
    cell_cols = [f"cell_rssi_{idx}" for idx in range(cfg.n_base_stations)]
    features = rf_feature_matrix(
        group[wifi_cols].to_numpy(dtype=np.float32),
        group[cell_cols].to_numpy(dtype=np.float32),
        group[CSI_COLUMNS].to_numpy(dtype=np.float32),
        cfg,
    )
    targets = group[["x", "y"]].to_numpy(dtype=np.float32)
    return features, targets


def build_sequences(df: pd.DataFrame, cfg: ProjectConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    episodes: list[int] = []
    for (episode, _), group in df.groupby(["episode", "user_id"], sort=True):
        features, targets = _user_features(group, cfg)
        max_start = len(group) - cfg.seq_len - cfg.pred_horizon + 1
        for start in range(max(0, max_start)):
            end = start + cfg.seq_len
            target_idx = end + cfg.pred_horizon - 1
            windows.append(features[start:end])
            labels.append(targets[target_idx])
            episodes.append(int(episode))

    if not windows:
        raise ValueError("No sequences were created; increase steps_per_episode or reduce seq_len/pred_horizon")

    return np.stack(windows).astype(np.float32), np.stack(labels).astype(np.float32), np.asarray(episodes)


def build_feature_splits(
    cfg: ProjectConfig,
    csv_path: str | Path,
    output_dir: str | Path,
) -> dict[str, np.ndarray]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    x, y, episodes = build_sequences(df, cfg)
    y = y / cfg.area_size

    # Split by episode: windows from one trajectory overlap heavily, so a random per-window
    # split would leak near-duplicate samples from train into test.
    rng = np.random.default_rng(cfg.seed)
    unique_eps = rng.permutation(np.unique(episodes))
    n_eps = len(unique_eps)
    n_val_eps = max(1, round(0.15 * n_eps))
    n_test_eps = max(1, round(0.15 * n_eps))
    n_train_eps = n_eps - n_val_eps - n_test_eps
    if n_train_eps < 1:
        raise ValueError("Need at least 3 episodes for non-empty train/val/test splits")
    split_eps = {
        "train": unique_eps[:n_train_eps],
        "val": unique_eps[n_train_eps : n_train_eps + n_val_eps],
        "test": unique_eps[n_train_eps + n_val_eps :],
    }
    idx = {name: rng.permutation(np.flatnonzero(np.isin(episodes, eps))) for name, eps in split_eps.items()}

    train_x = x[idx["train"]]
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x = ((x - mean) / std).astype(np.float32)
    y = y.astype(np.float32)

    splits = {}
    for name in ("train", "val", "test"):
        splits[f"X_{name}"] = x[idx[name]]
        splits[f"y_{name}"] = y[idx[name]]

    for name, arr in splits.items():
        np.save(output_dir / f"{name}.npy", arr.astype(np.float32))
    np.savez(
        output_dir / "normalization.npz",
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        area_size=np.float32(cfg.area_size),
        **{f"episodes_{name}": eps for name, eps in split_eps.items()},
    )
    return {name: arr.astype(np.float32) for name, arr in splits.items()}
