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
    assert cfg.coverage_radius == 150.0
    assert cfg.maddpg_warmup_steps == 16
    assert cfg.maddpg_update_every == 1
    assert cfg.maddpg_noise_start == 0.30
    assert cfg.maddpg_noise_end == 0.05


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
    assert cfg.coverage_radius == 170.0
    assert cfg.battery_drain_rate == 0.008
    assert cfg.position_source == "predicted"
    assert cfg.predictor_epochs == 80
    assert cfg.maddpg_warmup_steps == 2000


def test_invalid_config_name_raises_value_error():
    try:
        get_config("wrong")
    except ValueError as exc:
        assert "Unknown config preset" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_apply_overrides_converts_types():
    from wifi_uav.config import apply_overrides

    cfg = apply_overrides(get_config("smoke"), ["maddpg_gamma=0.9", "maddpg_updates_per_step=2"])

    assert cfg.maddpg_gamma == 0.9
    assert cfg.maddpg_updates_per_step == 2
