from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import numpy as np

from wifi_uav.config import apply_overrides, get_config
from wifi_uav.device import best_device
from wifi_uav.evaluate import plot_predictor_history
from wifi_uav.predictor import center_guess_rmse_meters, rmse_meters, train_predictor_on_arrays


def log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LSTM-Transformer position predictor.")
    parser.add_argument("--preset", default="smoke", choices=["smoke", "medium", "full"])
    parser.add_argument("--epochs", type=int, default=None, help="defaults to the preset's predictor_epochs")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="demo mode: train and print live, but keep existing checkpoints and results untouched",
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any config field")
    args = parser.parse_args()

    cfg = apply_overrides(get_config(args.preset), args.set)
    device = best_device(allow_mps=True)
    x_train = np.load(cfg.data_dir / "X_train.npy")
    y_train = np.load(cfg.data_dir / "y_train.npy")
    x_val = np.load(cfg.data_dir / "X_val.npy")
    y_val = np.load(cfg.data_dir / "y_val.npy")
    x_test = np.load(cfg.data_dir / "X_test.npy")
    y_test = np.load(cfg.data_dir / "y_test.npy")
    log(f"Input windows: {x_train.shape[1]} steps x {x_train.shape[2]} features; predicting position {cfg.pred_horizon} steps ahead")

    scratch = Path(tempfile.mkdtemp()) if args.no_save else None
    checkpoint = (scratch or cfg.checkpoint_dir) / "predictor_best.pt"
    if args.no_save:
        log("Demo mode (--no-save): existing checkpoints and results will not be changed.")

    history, model = train_predictor_on_arrays(
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=args.epochs if args.epochs is not None else cfg.predictor_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.learning_rate,
        device=device,
        checkpoint_path=checkpoint,
        area_size=cfg.area_size,
        log=log,
    )
    val_rmse = rmse_meters(model, x_val, y_val, cfg.area_size, device=device)
    rmse = rmse_meters(model, x_test, y_test, cfg.area_size, device=device)
    center_rmse = center_guess_rmse_meters(y_test, cfg.area_size)
    best_epoch = int(np.argmin(history["val_loss"])) + 1

    log("")
    log(f"Best epoch: {best_epoch} of {len(history['train_loss'])} run")
    log(f"Validation RMSE: {val_rmse:.2f} m")
    log(f"Test RMSE (held-out episodes): {rmse:.2f} m  |  centre-guess baseline: {center_rmse:.2f} m")
    if args.no_save:
        return

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "predictor_summary.txt").write_text(
        f"RMSE_meters={rmse:.4f}\nVal_RMSE_meters={val_rmse:.4f}\n"
        f"Center_guess_RMSE_meters={center_rmse:.4f}\nepochs_run={len(history['train_loss'])}\n"
        f"best_epoch={best_epoch}\n",
        encoding="utf-8",
    )
    with (cfg.result_dir / "predictor_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *history])
        for epoch, row in enumerate(zip(*history.values()), start=1):
            writer.writerow([epoch, *(f"{value:.6g}" for value in row)])
    plot_predictor_history(history, cfg.result_dir / "predictor_training.png")
    log(f"Saved {cfg.result_dir / 'predictor_history.csv'} and {cfg.result_dir / 'predictor_training.png'}")


if __name__ == "__main__":
    main()
