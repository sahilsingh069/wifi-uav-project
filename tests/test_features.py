import numpy as np

from wifi_uav.config import get_config
from wifi_uav.features import build_feature_splits
from wifi_uav.signal_model import generate_rf_dataset


def test_build_feature_splits_shapes_and_files(tmp_path):
    cfg = get_config("smoke")
    csv_path = tmp_path / "rf_dataset.csv"
    out_dir = tmp_path / "features"
    generate_rf_dataset(cfg, csv_path)

    splits = build_feature_splits(cfg, csv_path, out_dir)

    assert splits["X_train"].ndim == 3
    assert splits["X_train"].shape[1] == cfg.seq_len
    assert splits["X_train"].shape[2] == cfg.n_aps * 3 + cfg.n_base_stations + 4 + 2
    assert splits["y_train"].shape[1] == 2
    assert len(splits["X_train"]) > 0
    assert len(splits["X_val"]) > 0
    assert len(splits["X_test"]) > 0
    assert splits["X_train"].dtype == np.float32
    assert splits["y_train"].dtype == np.float32
    assert np.isfinite(splits["X_train"]).all()
    assert np.isfinite(splits["y_train"]).all()
    assert (out_dir / "X_train.npy").exists()
    assert (out_dir / "normalization.npz").exists()


def test_splits_are_disjoint_by_episode(tmp_path):
    cfg = get_config("smoke")
    csv_path = tmp_path / "rf_dataset.csv"
    generate_rf_dataset(cfg, csv_path)

    build_feature_splits(cfg, csv_path, tmp_path)

    norm = np.load(tmp_path / "normalization.npz")
    train, val, test = (set(norm[f"episodes_{name}"].tolist()) for name in ("train", "val", "test"))
    assert train and val and test
    assert not (train & val) and not (train & test) and not (val & test)
