import numpy as np

from wifi_uav.config import get_config
from wifi_uav.evaluate import moving_average, run_baseline_policy, write_summary
from wifi_uav.uav_env import UAVCoverageEnv


def test_moving_average_handles_short_arrays():
    result = moving_average(np.array([1.0, 2.0, 3.0]), window=20)

    assert result.shape == (3,)
    assert np.isclose(result[-1], 2.0)


def test_baseline_policy_returns_coverage():
    cfg = get_config("smoke")
    env = UAVCoverageEnv(cfg, seed=cfg.seed)

    score = run_baseline_policy(env, "static", episodes=2)

    assert 0.0 <= score <= 1.0


def test_write_summary_creates_file(tmp_path):
    path = tmp_path / "summary.txt"

    write_summary(path, predictor_rmse=8.5, maddpg_coverage=0.82, greedy_coverage=0.71, static_coverage=0.44)

    text = path.read_text(encoding="utf-8")
    assert "Predictor RMSE" in text
    assert "MADDPG coverage" in text
