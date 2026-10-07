import torch

from wifi_uav.predictor import LSTMTransformerPredictor, train_predictor_on_arrays


def test_lstm_transformer_output_shape_and_range():
    model = LSTMTransformerPredictor(input_dim=31)
    x = torch.randn(4, 10, 31)

    y = model(x)

    assert y.shape == (4, 2)
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)


def test_short_training_returns_history():
    x = torch.randn(24, 10, 31).numpy()
    y = torch.rand(24, 2).numpy()

    history, model = train_predictor_on_arrays(x, y, x, y, epochs=2, batch_size=8, device="cpu")

    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert isinstance(model, LSTMTransformerPredictor)


def test_online_predictor_drives_env_observations(tmp_path):
    import numpy as np

    from wifi_uav.config import get_config
    from wifi_uav.features import build_feature_splits
    from wifi_uav.predictor import OnlinePositionPredictor
    from wifi_uav.signal_model import generate_rf_dataset
    from wifi_uav.uav_env import UAVCoverageEnv

    cfg = get_config("smoke")
    csv_path = tmp_path / "rf.csv"
    generate_rf_dataset(cfg, csv_path)
    splits = build_feature_splits(cfg, csv_path, tmp_path)
    checkpoint = tmp_path / "predictor.pt"
    train_predictor_on_arrays(
        splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"],
        epochs=1, batch_size=32, checkpoint_path=checkpoint,
    )
    predictor = OnlinePositionPredictor.from_files(cfg, checkpoint, tmp_path / "normalization.npz")
    env = UAVCoverageEnv(cfg, seed=1, predictor=predictor)

    env.reset()
    _, _, _, info = env.step([np.zeros(3, dtype=np.float32) for _ in range(cfg.n_uavs)])

    assert env.observed_user_positions.shape == (cfg.n_users, 2)
    assert np.all((env.observed_user_positions >= 0) & (env.observed_user_positions <= cfg.area_size))
    assert info["prediction_error_m"] > 0.0
