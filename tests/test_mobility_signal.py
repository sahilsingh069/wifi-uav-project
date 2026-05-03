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
    assert sample["csi_phase_mean"].shape == (cfg.n_users,)
    assert sample["los_flags"].shape == (cfg.n_users, cfg.n_aps)


def test_generate_rf_dataset_has_manual_columns(tmp_path):
    cfg = get_config("smoke")
    out = tmp_path / "rf_dataset.csv"
    df = generate_rf_dataset(cfg, out)
    assert out.exists()
    assert len(df) == cfg.episodes * cfg.steps_per_episode * cfg.n_users
    required = {"episode", "step", "user_id", "x", "y", "wifi_rssi_0", "cell_rssi_0", "csi_amp_mean", "csi_phase_mean"}
    assert required.issubset(df.columns)
    loaded = pd.read_csv(out)
    assert len(loaded) == len(df)
