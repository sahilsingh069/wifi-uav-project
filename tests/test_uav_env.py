import numpy as np

from wifi_uav.config import get_config
from wifi_uav.uav_env import UAVCoverageEnv


def test_env_reset_and_step_shapes():
    cfg = get_config("smoke")
    env = UAVCoverageEnv(cfg, seed=cfg.seed)
    obs = env.reset()

    assert len(obs) == cfg.n_uavs
    assert obs[0].ndim == 1
    assert len(obs[0]) == env.obs_dim

    actions = [np.zeros(3, dtype=np.float32) for _ in range(cfg.n_uavs)]
    next_obs, reward, done, info = env.step(actions)

    assert len(next_obs) == cfg.n_uavs
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert 0.0 <= info["coverage"] <= 1.0


def test_env_clips_uav_positions_and_altitudes():
    cfg = get_config("smoke")
    env = UAVCoverageEnv(cfg, seed=cfg.seed)
    env.reset()
    actions = [np.array([99.0, 99.0, 99.0], dtype=np.float32) for _ in range(cfg.n_uavs)]

    env.step(actions)

    assert env.uav_positions[:, 0].max() <= cfg.area_size
    assert env.uav_positions[:, 1].max() <= cfg.area_size
    assert env.uav_positions[:, 2].max() <= cfg.uav_alt_max
