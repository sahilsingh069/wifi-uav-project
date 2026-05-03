from wifi_uav.config import ProjectConfig, get_config


def test_smoke_config_is_small_and_valid():
    cfg = get_config("smoke")
    assert cfg.area_size == 500.0
    assert cfg.n_users == 6
    assert cfg.n_aps == 8
    assert cfg.n_base_stations == 3
    assert cfg.seq_len == 10
    assert cfg.pred_horizon == 5
    assert cfg.n_uavs == 3
    assert cfg.episodes == 3
    assert cfg.steps_per_episode == 25


def test_full_config_matches_manual_core_values():
    cfg = get_config("full")
    assert cfg.n_users == 30
    assert cfg.n_aps == 8
    assert cfg.n_base_stations == 3
    assert cfg.n_uavs == 5
    assert cfg.episodes == 500
    assert cfg.steps_per_episode == 150
    assert cfg.drl_episodes == 500
    assert cfg.drl_episode_len == 100


def test_invalid_config_name_raises_value_error():
    try:
        get_config("wrong")
    except ValueError as exc:
        assert "Unknown config preset" in str(exc)
    else:
        raise AssertionError("expected ValueError")
