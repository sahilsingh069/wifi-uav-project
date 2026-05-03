from dataclasses import replace

import pandas as pd

from wifi_uav.config import get_config
from wifi_uav.mobility import GaussMarkovMobility
from wifi_uav.signal_model import PhysicsSignalModel, generate_rf_dataset


def test_gauss_markov_positions_stay_inside_area():
    cfg = get_config("smoke")
    mobility = GaussMarkovMobility(cfg, seed=cfg.seed)
    positions = mobility.reset()
    for _ in range(20):
        positions = mobility.step()
        assert positions.shape == (cfg.n_users, 2)
        assert positions.min() >= 0.0
        assert positions.max() <= cfg.area_size


def test_signal_model_outputs_expected_keys():
    cfg = get_config("smoke")
    model = PhysicsSignalModel(cfg, seed=cfg.seed)
    user_xy = model.rng.uniform(0.0, cfg.area_size, size=(cfg.n_users, 2))
    sample = model.measure(user_xy)
    assert sample["wifi_rssi"].shape == (cfg.n_users, cfg.n_aps)
    assert sample["cell_rssi"].shape == (cfg.n_users, cfg.n_base_stations)
    assert sample["csi_amp_mean"].shape == (cfg.n_users,)
    assert sample["csi_amp_std"].shape == (cfg.n_users,)
    assert sample["csi_amp_max"].shape == (cfg.n_users,)
    assert sample["csi_phase_mean"].shape == (cfg.n_users,)
    assert sample["los_flags"].shape == (cfg.n_users, cfg.n_aps)


def test_generate_rf_dataset_has_manual_columns(tmp_path):
    cfg = get_config("smoke")
    out = tmp_path / "rf_dataset.csv"
    df = generate_rf_dataset(cfg, out)
    assert out.exists()
    assert len(df) == cfg.episodes * cfg.steps_per_episode * cfg.n_users
    required = {
        "episode",
        "step",
        "user_id",
        "x",
        "y",
        "csi_amp_mean",
        "csi_amp_std",
        "csi_amp_max",
        "csi_phase_mean",
    }
    required.update(f"wifi_rssi_{ap_idx}" for ap_idx in range(cfg.n_aps))
    required.update(f"los_ap_{ap_idx}" for ap_idx in range(cfg.n_aps))
    required.update(f"cell_rssi_{bs_idx}" for bs_idx in range(cfg.n_base_stations))
    assert required.issubset(df.columns)
    loaded = pd.read_csv(out)
    assert len(loaded) == len(df)


def test_generate_rf_dataset_supports_custom_transmitter_counts(tmp_path):
    cfg = replace(get_config("smoke"), n_aps=9, n_base_stations=4)
    out = tmp_path / "rf_dataset.csv"

    df = generate_rf_dataset(cfg, out)

    assert out.exists()
    assert {"wifi_rssi_8", "los_ap_8", "cell_rssi_3"}.issubset(df.columns)
    assert len(df) == cfg.episodes * cfg.steps_per_episode * cfg.n_users
