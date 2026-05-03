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
