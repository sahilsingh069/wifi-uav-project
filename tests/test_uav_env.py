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


def test_altitude_changes_footprint_with_interior_optimum():
    cfg = get_config("full")
    env = UAVCoverageEnv(cfg, seed=0)
    best = env.optimal_altitude

    assert cfg.uav_alt_min < best < cfg.uav_alt_max
    assert np.isclose(env.footprint_radius(best), cfg.coverage_radius, atol=1.0)
    assert env.footprint_radius(cfg.uav_alt_min) < cfg.coverage_radius
    assert env.footprint_radius(cfg.uav_alt_max) < cfg.coverage_radius


def test_empty_battery_grounds_uav():
    cfg = get_config("smoke")
    env = UAVCoverageEnv(cfg, seed=cfg.seed)
    env.reset()
    env.battery[0] = 0.0
    before = env.uav_positions[0].copy()

    env.step([np.ones(3, dtype=np.float32) for _ in range(cfg.n_uavs)])

    assert np.allclose(env.uav_positions[0], before)
    env.battery[:] = 0.0
    assert env._compute_coverage() == 0.0


def test_oracle_mode_observes_true_positions():
    cfg = get_config("smoke")
    env = UAVCoverageEnv(cfg, seed=cfg.seed)
    env.reset()
    _, _, _, info = env.step([np.zeros(3, dtype=np.float32) for _ in range(cfg.n_uavs)])

    assert info["prediction_error_m"] == 0.0
