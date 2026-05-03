from __future__ import annotations

import argparse

import numpy as np
import torch

from wifi_uav.config import get_config
from wifi_uav.predictor import rmse_meters, train_predictor_on_arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="smoke", choices=["smoke", "medium", "full"])
    parser.add_argument("--epochs", type=int, default=2)
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
        epochs=args.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.learning_rate,
        device=device,
        checkpoint_path=cfg.checkpoint_dir / "predictor_best.pt",
    )
    rmse = rmse_meters(model, x_test, y_test, cfg.area_size, device=device)
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "predictor_summary.txt").write_text(f"RMSE_meters={rmse:.4f}\n", encoding="utf-8")
    print(f"Train losses: {history['train_loss']}")
    print(f"Mean RMSE: {rmse:.2f} m")


if __name__ == "__main__":
    main()
