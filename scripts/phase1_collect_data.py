from __future__ import annotations

import argparse

from wifi_uav.config import get_config
from wifi_uav.signal_model import generate_rf_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the phase 1 RF mobility dataset.")
    parser.add_argument("--preset", choices=["smoke", "medium", "full"], default="smoke")
    args = parser.parse_args()

    cfg = get_config(args.preset)
    output = cfg.data_dir / "rf_dataset.csv"
    df = generate_rf_dataset(cfg, output)
    print(f"Saved RF dataset: {output}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
