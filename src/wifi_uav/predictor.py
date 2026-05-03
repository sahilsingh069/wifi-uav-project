from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMTransformerPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int = 128,
        n_heads: int = 4,
        n_tf_layers: int = 2,
        n_lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.lstm = nn.LSTM(
            hidden,
            hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_tf_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        _, (h_n, _) = self.lstm(z)
        lstm_feat = h_n[-1]
        tf_feat = self.transformer(z).mean(dim=1)
        return self.head(torch.cat([lstm_feat, tf_feat], dim=1))


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        total += float(loss.item()) * len(xb)
        count += len(xb)
    return total / max(1, count)


def train_predictor_on_arrays(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    checkpoint_path: str | Path | None = None,
) -> tuple[dict[str, list[float]], LSTMTransformerPredictor]:
    device = torch.device(device)
    model = LSTMTransformerPredictor(input_dim=x_train.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    loss_fn = nn.HuberLoss()
    train_loader = _loader(x_train, y_train, batch_size, True)
    val_loader = _loader(x_val, y_val, batch_size, False)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    patience = 15

    for _ in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)

        scheduler.step()
        train_loss = total / max(1, count)
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
            if checkpoint_path is not None:
                checkpoint_path = Path(checkpoint_path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": best_state, "history": history}, checkpoint_path)
        else:
            stale += 1

        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, model


@torch.no_grad()
def rmse_meters(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    area_size: float,
    device: str | torch.device = "cpu",
) -> float:
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    pred = model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
    err = (pred - y) * area_size
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
