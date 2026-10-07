from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from wifi_uav.config import ProjectConfig


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
                torch.save(
                    {"model": best_state, "history": history, "input_dim": int(x_train.shape[-1])},
                    checkpoint_path,
                )
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


def center_guess_rmse_meters(y: np.ndarray, area_size: float) -> float:
    """RMSE of always predicting the area centre: the floor any useful predictor must beat."""
    err = (np.asarray(y) - 0.5) * area_size
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def load_predictor(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> LSTMTransformerPredictor:
    state = torch.load(checkpoint_path, map_location=device)
    weights = state["model"]
    input_dim = int(state.get("input_dim", weights["input_proj.0.weight"].shape[1]))
    model = LSTMTransformerPredictor(input_dim=input_dim).to(device)
    model.load_state_dict(weights)
    model.eval()
    return model


class OnlinePositionPredictor:
    """Turns a live stream of RF measurements into predicted user positions.

    Keeps a short rolling history of raw measurements per user, rebuilds the same feature
    windows used in training, and returns the predicted (x, y) ``pred_horizon`` steps ahead.
    """

    def __init__(
        self,
        cfg: "ProjectConfig",
        model: LSTMTransformerPredictor,
        mean: np.ndarray,
        std: np.ndarray,
        device: str | torch.device = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
        # Two extra rows so delta/rolling-variance at the window start match offline features.
        self.history_len = cfg.seq_len + 2
        self.history: list[dict[str, np.ndarray]] = []

    @classmethod
    def from_files(
        cls,
        cfg: "ProjectConfig",
        checkpoint_path: str | Path,
        normalization_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "OnlinePositionPredictor":
        norm = np.load(normalization_path)
        return cls(cfg, load_predictor(checkpoint_path, device), norm["mean"], norm["std"], device)

    def reset(self) -> None:
        self.history = []

    def observe(self, sample: dict[str, np.ndarray]) -> None:
        self.history.append(
            {
                "wifi": np.asarray(sample["wifi_rssi"], dtype=np.float32),
                "cell": np.asarray(sample["cell_rssi"], dtype=np.float32),
                "csi": np.column_stack(
                    [sample[name] for name in ("csi_amp_mean", "csi_amp_std", "csi_amp_max", "csi_phase_mean")]
                ).astype(np.float32),
            }
        )
        self.history = self.history[-self.history_len :]

    @torch.no_grad()
    def predict(self) -> np.ndarray:
        from wifi_uav.features import rf_feature_matrix

        if not self.history:
            raise RuntimeError("observe() must be called before predict()")
        wifi = np.stack([h["wifi"] for h in self.history], axis=1)
        cell = np.stack([h["cell"] for h in self.history], axis=1)
        csi = np.stack([h["csi"] for h in self.history], axis=1)
        windows = []
        for user in range(wifi.shape[0]):
            feats = rf_feature_matrix(wifi[user], cell[user], csi[user], self.cfg)[-self.cfg.seq_len :]
            if len(feats) < self.cfg.seq_len:
                # Episode start: pad by repeating the earliest step until a full window exists.
                pad = np.repeat(feats[:1], self.cfg.seq_len - len(feats), axis=0)
                feats = np.concatenate([pad, feats], axis=0)
            windows.append(feats)
        x = (np.stack(windows) - self.mean) / self.std
        pred = self.model(torch.tensor(x, dtype=torch.float32, device=self.device)).cpu().numpy()
        return (pred * self.cfg.area_size).astype(np.float32)
