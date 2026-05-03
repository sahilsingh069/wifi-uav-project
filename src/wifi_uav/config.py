from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    area_size: float = 500.0
    uav_alt_min: float = 50.0
    uav_alt_max: float = 150.0
    n_users: int = 30
    user_speed: float = 1.5
    n_aps: int = 8
    ap_height: float = 5.0
    tx_power_dbm: float = 20.0
    wifi_freq_ghz: float = 2.4
    n_base_stations: int = 3
    bs_height: float = 25.0
    cell_freq_ghz: float = 3.5
    episodes: int = 500
    steps_per_episode: int = 150
    dt: float = 0.5
    seq_len: int = 10
    pred_horizon: int = 5
    n_uavs: int = 5
    v_max: float = 20.0
    batch_size: int = 256
    learning_rate: float = 1e-3
    drl_episodes: int = 500
    drl_episode_len: int = 100
    seed: int = 42
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    checkpoint_dir: Path = Path("checkpoints")
    result_dir: Path = Path("results")


def get_config(preset: str = "smoke") -> ProjectConfig:
    if preset == "smoke":
        return ProjectConfig(
            n_users=6,
            episodes=3,
            steps_per_episode=25,
            n_uavs=3,
            batch_size=32,
            drl_episodes=3,
            drl_episode_len=12,
        )
    if preset == "medium":
        return ProjectConfig(
            n_users=18,
            episodes=80,
            steps_per_episode=100,
            n_uavs=5,
            batch_size=128,
            drl_episodes=80,
            drl_episode_len=60,
        )
    if preset == "full":
        return ProjectConfig()
    raise ValueError(f"Unknown config preset: {preset}")
