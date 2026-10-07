from __future__ import annotations

import argparse

import numpy as np
import torch

from wifi_uav.config import get_config
from wifi_uav.predictor import center_guess_rmse_meters, rmse_meters, train_predictor_on_arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="smoke", choices=["smoke", "medium", "full"])
    parser.add_argument("--epochs", type=int, default=None, help="defaults to the preset's predictor_epochs")
    args = parser.parse_args()

    cfg = get_config(args.preset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train = np.load(cfg.data_dir / "X_train.npy")
    y_train = np.load(cfg.data_dir / "y_train.npy")
    x_val = np.load(cfg.data_dir / "X_val.npy")
    y_val = np.load(cfg.data_dir / "y_val.npy")
    x_test = np.load(cfg.data_dir / "X_test.npy")
    y_test = np.load(cfg.data_dir / "y_test.npy")

    history, model = train_predictor_on_arrays(
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=args.epochs if args.epochs is not None else cfg.predictor_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.learning_rate,
        device=device,
        checkpoint_path=cfg.checkpoint_dir / "predictor_best.pt",
    )
    rmse = rmse_meters(model, x_test, y_test, cfg.area_size, device=device)
    center_rmse = center_guess_rmse_meters(y_test, cfg.area_size)
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "predictor_summary.txt").write_text(
        f"RMSE_meters={rmse:.4f}\nCenter_guess_RMSE_meters={center_rmse:.4f}\nepochs_run={len(history['train_loss'])}\n",
        encoding="utf-8",
    )
    print(f"Epochs run: {len(history['train_loss'])}, final train loss: {history['train_loss'][-1]:.5f}")
    print(f"Test RMSE (held-out episodes): {rmse:.2f} m  |  centre-guess baseline: {center_rmse:.2f} m")


if __name__ == "__main__":
    main()
