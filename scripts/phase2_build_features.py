from __future__ import annotations

import argparse

from wifi_uav.config import get_config
from wifi_uav.features import build_feature_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="smoke", choices=["smoke", "medium", "full"])
    args = parser.parse_args()

    cfg = get_config(args.preset)
    splits = build_feature_splits(cfg, cfg.data_dir / "rf_dataset.csv", cfg.data_dir)
    print(f"X_train: {splits['X_train'].shape}")
    print(f"y_train: {splits['y_train'].shape}")
    print("Feature splits saved.")


if __name__ == "__main__":
    main()
