# Wi-Fi UAV Full Project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete Colab-ready college project for Wi-Fi/cellular user mobility prediction and multi-UAV trajectory optimization using a Sionna-ready RF simulator, LSTM-Transformer predictor, MADDPG trainer, evaluation plots, report assets, and presentation assets.

**Architecture:** The project is a modular Python package with phase scripts and one Colab notebook that mirror the build manual. Heavy training runs in Google Colab GPU, while source code, tests, report assets, and reproducible scripts live in this repository. The first working backend uses a physics-based 3GPP-style RF simulator with a clean interface that can later be replaced by a real Sionna RT backend.

**Tech Stack:** Python 3.10+, NumPy, pandas, scikit-learn, PyTorch, gymnasium-style environment interfaces, matplotlib, seaborn, pytest, nbformat.

---

## File Structure

- Create: `pyproject.toml` - package metadata, pytest config, runtime dependencies
- Create: `README.md` - project purpose, Colab workflow, phase commands
- Create: `src/wifi_uav/__init__.py` - package marker and version
- Create: `src/wifi_uav/config.py` - dataclass configuration and smoke/full presets
- Create: `src/wifi_uav/mobility.py` - Gauss-Markov mobility model
- Create: `src/wifi_uav/signal_model.py` - RF signal simulator and dataset writer
- Create: `src/wifi_uav/features.py` - feature extraction, normalization, splits
- Create: `src/wifi_uav/predictor.py` - LSTM-Transformer model and training utilities
- Create: `src/wifi_uav/uav_env.py` - multi-UAV coverage environment
- Create: `src/wifi_uav/maddpg.py` - replay buffer, actor/critic, MADDPG agents
- Create: `src/wifi_uav/evaluate.py` - metrics, baselines, plots, summaries
- Create: `scripts/phase1_collect_data.py` - RF data generation entrypoint
- Create: `scripts/phase2_build_features.py` - feature dataset entrypoint
- Create: `scripts/phase3_train_predictor.py` - predictor training entrypoint
- Create: `scripts/phase4_sanity_env.py` - UAV environment sanity check
- Create: `scripts/phase5_train_maddpg.py` - MADDPG training entrypoint
- Create: `scripts/phase6_evaluate.py` - evaluation and artifact entrypoint
- Create: `notebooks/WiFi_UAV_Full_Project_Colab.ipynb` - executable Colab notebook
- Create: `reports/project_report_outline.md` - report scaffold
- Create: `reports/presentation_outline.md` - PPT scaffold
- Create: `tests/test_config.py` - configuration tests
- Create: `tests/test_mobility_signal.py` - mobility and RF simulator tests
- Create: `tests/test_features.py` - feature extraction tests
- Create: `tests/test_predictor.py` - model shape and short training tests
- Create: `tests/test_uav_env.py` - environment reset/step/reward tests
- Create: `tests/test_maddpg.py` - replay buffer and network tests
- Create: `tests/test_evaluate.py` - baseline/plot tests

---

### Task 1: Project Foundation

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/wifi_uav/__init__.py`
- Create: `src/wifi_uav/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write config tests**

Create `tests/test_config.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'wifi_uav'`.

- [ ] **Step 3: Add package metadata**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "wifi-uav-project"
version = "0.1.0"
description = "Wi-Fi/cellular mobility prediction and UAV trajectory optimization college project"
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.24",
  "pandas>=2.0",
  "scikit-learn>=1.3",
  "torch>=2.1",
  "matplotlib>=3.7",
  "seaborn>=0.13",
  "tqdm>=4.66",
  "gymnasium>=0.29",
  "nbformat>=5.9"
]

[project.optional-dependencies]
dev = ["pytest>=7.4"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 4: Add configuration module**

Create `src/wifi_uav/__init__.py`:

```python
"""Wi-Fi/cellular UAV mobility prediction and trajectory optimization project."""

__version__ = "0.1.0"
```

Create `src/wifi_uav/config.py`:

```python
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
```

- [ ] **Step 5: Add README**

Create `README.md`:

```markdown
# Wi-Fi/Cellular UAV Mobility Prediction and Trajectory Optimization

College project implementing a full RF-signal-to-UAV-control research pipeline:

1. Generate Wi-Fi/cellular RF mobility data
2. Extract time-series signal features
3. Train an LSTM-Transformer mobility predictor
4. Build a multi-UAV coverage environment
5. Train MADDPG agents
6. Evaluate against greedy, random, and static baselines

## Recommended Runtime

Use Google Colab with GPU enabled for training. Local execution is intended for smoke tests and development.

## Quick Local Smoke Test

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Phase Scripts

```bash
python scripts/phase1_collect_data.py --preset smoke
python scripts/phase2_build_features.py --preset smoke
python scripts/phase3_train_predictor.py --preset smoke --epochs 2
python scripts/phase4_sanity_env.py --preset smoke
python scripts/phase5_train_maddpg.py --preset smoke
python scripts/phase6_evaluate.py --preset smoke
```
```

- [ ] **Step 6: Run foundation tests**

Run:

```bash
python -m pip install -e ".[dev]"
pytest tests/test_config.py -v
```

Expected: PASS with `3 passed`.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml README.md src/wifi_uav/__init__.py src/wifi_uav/config.py tests/test_config.py
git commit -m "feat: add project foundation"
```

---

### Task 2: Mobility and RF Signal Dataset Generation

**Files:**
- Create: `src/wifi_uav/mobility.py`
- Create: `src/wifi_uav/signal_model.py`
- Create: `scripts/phase1_collect_data.py`
- Create: `tests/test_mobility_signal.py`

- [ ] **Step 1: Write mobility and signal tests**

Create `tests/test_mobility_signal.py`:

```python
import pandas as pd

from wifi_uav.config import get_config
from wifi_uav.mobility import GaussMarkovMobility
from wifi_uav.signal_model import PhysicsSignalModel, generate_rf_dataset


def test_gauss_markov_positions_stay_inside_area():
    cfg = get_config("smoke")
    mobility = GaussMarkovMobility(cfg, seed=cfg.seed)
    positions = mobility.reset()
    for _ in range(20):
        positions = mobility.step()
        assert positions.shape == (cfg.n_users, 2)
        assert positions.min() >= 0.0
        assert positions.max() <= cfg.area_size


def test_signal_model_outputs_expected_keys():
    cfg = get_config("smoke")
    model = PhysicsSignalModel(cfg, seed=cfg.seed)
    user_xy = model.rng.uniform(0.0, cfg.area_size, size=(cfg.n_users, 2))
    sample = model.measure(user_xy)
    assert sample["wifi_rssi"].shape == (cfg.n_users, cfg.n_aps)
    assert sample["cell_rssi"].shape == (cfg.n_users, cfg.n_base_stations)
    assert sample["csi_amp_mean"].shape == (cfg.n_users,)
    assert sample["csi_phase_mean"].shape == (cfg.n_users,)
    assert sample["los_flags"].shape == (cfg.n_users, cfg.n_aps)


def test_generate_rf_dataset_has_manual_columns(tmp_path):
    cfg = get_config("smoke")
    out = tmp_path / "rf_dataset.csv"
    df = generate_rf_dataset(cfg, out)
    assert out.exists()
    assert len(df) == cfg.episodes * cfg.steps_per_episode * cfg.n_users
    required = {"episode", "step", "user_id", "x", "y", "wifi_rssi_0", "cell_rssi_0", "csi_amp_mean", "csi_phase_mean"}
    assert required.issubset(df.columns)
    loaded = pd.read_csv(out)
    assert len(loaded) == len(df)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_mobility_signal.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `wifi_uav.mobility`.

- [ ] **Step 3: Implement mobility**

Create `src/wifi_uav/mobility.py`:

```python
from __future__ import annotations

import numpy as np

from wifi_uav.config import ProjectConfig


class GaussMarkovMobility:
    def __init__(self, cfg: ProjectConfig, alpha: float = 0.75, seed: int | None = None):
        self.cfg = cfg
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.positions = np.zeros((cfg.n_users, 2), dtype=np.float32)
        self.velocities = np.zeros((cfg.n_users, 2), dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.positions = self.rng.uniform(0.0, self.cfg.area_size, size=(self.cfg.n_users, 2)).astype(np.float32)
        angles = self.rng.uniform(0.0, 2.0 * np.pi, size=self.cfg.n_users)
        speeds = self.rng.normal(self.cfg.user_speed, 0.2, size=self.cfg.n_users).clip(0.2, 3.0)
        self.velocities = np.column_stack([np.cos(angles) * speeds, np.sin(angles) * speeds]).astype(np.float32)
        return self.positions.copy()

    def step(self) -> np.ndarray:
        noise = self.rng.normal(0.0, 0.35, size=(self.cfg.n_users, 2))
        target = self.rng.normal(0.0, self.cfg.user_speed, size=(self.cfg.n_users, 2))
        self.velocities = (self.alpha * self.velocities + (1.0 - self.alpha) * target + noise).astype(np.float32)
        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True).clip(1e-6)
        self.velocities = self.velocities / speeds * np.minimum(speeds, self.cfg.user_speed * 2.0)
        self.positions = self.positions + self.velocities * self.cfg.dt
        for dim in range(2):
            low = self.positions[:, dim] < 0.0
            high = self.positions[:, dim] > self.cfg.area_size
            self.positions[low, dim] = 0.0
            self.positions[high, dim] = self.cfg.area_size
            self.velocities[low | high, dim] *= -0.7
        return self.positions.copy()
```

- [ ] **Step 4: Implement RF signal model**

Create `src/wifi_uav/signal_model.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wifi_uav.config import ProjectConfig
from wifi_uav.mobility import GaussMarkovMobility


def place_access_points(cfg: ProjectConfig) -> np.ndarray:
    xs = np.linspace(70.0, cfg.area_size - 70.0, 4)
    ys = np.linspace(110.0, cfg.area_size - 110.0, 2)
    return np.array([[x, y, cfg.ap_height] for y in ys for x in xs], dtype=np.float32)[: cfg.n_aps]


def place_base_stations(cfg: ProjectConfig) -> np.ndarray:
    points = [
        [50.0, 50.0, cfg.bs_height],
        [cfg.area_size - 50.0, 60.0, cfg.bs_height],
        [cfg.area_size / 2.0, cfg.area_size - 60.0, cfg.bs_height],
    ]
    return np.array(points, dtype=np.float32)[: cfg.n_base_stations]


class PhysicsSignalModel:
    def __init__(self, cfg: ProjectConfig, seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.ap_positions = place_access_points(cfg)
        self.bs_positions = place_base_stations(cfg)

    def _path_loss_db(self, distances: np.ndarray, freq_ghz: float, los: np.ndarray) -> np.ndarray:
        d = np.maximum(distances, 1.0)
        fspl = 32.4 + 20.0 * np.log10(freq_ghz) + 20.0 * np.log10(d)
        nlos_extra = np.where(los, 0.0, 18.0 + 8.0 * np.log10(d))
        return fspl + nlos_extra

    def _los_probability(self, distances: np.ndarray) -> np.ndarray:
        d = np.maximum(distances, 1.0)
        return np.minimum(18.0 / d, 1.0) * (1.0 - np.exp(-d / 63.0)) + np.exp(-d / 63.0)

    def _measure_links(self, user_xy: np.ndarray, tx_xyz: np.ndarray, freq_ghz: float) -> tuple[np.ndarray, np.ndarray]:
        user_xyz = np.column_stack([user_xy, np.full(len(user_xy), 1.5)])
        diff = user_xyz[:, None, :] - tx_xyz[None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        los_prob = self._los_probability(distances)
        los = self.rng.random(distances.shape) < los_prob
        shadowing = self.rng.normal(0.0, 4.0, size=distances.shape)
        fading = self.rng.rayleigh(scale=1.0, size=distances.shape)
        fading_db = 20.0 * np.log10(np.maximum(fading, 1e-3))
        rssi = self.cfg.tx_power_dbm - self._path_loss_db(distances, freq_ghz, los) + shadowing + fading_db
        return rssi.astype(np.float32), los

    def _csi_summary(self, n_users: int) -> dict[str, np.ndarray]:
        n_subcarriers = 52
        n_paths = self.rng.integers(3, 6, size=n_users)
        amp = np.zeros((n_users, n_subcarriers), dtype=np.float32)
        phase = np.zeros((n_users, n_subcarriers), dtype=np.float32)
        for user in range(n_users):
            gains = self.rng.rayleigh(scale=1.0, size=n_paths[user])
            phases = self.rng.uniform(-np.pi, np.pi, size=n_paths[user])
            delays = self.rng.exponential(scale=0.4, size=n_paths[user])
            sub = np.arange(n_subcarriers)
            response = np.zeros(n_subcarriers, dtype=np.complex64)
            for gain, ph, delay in zip(gains, phases, delays):
                response += gain * np.exp(1j * (ph - 2.0 * np.pi * delay * sub / n_subcarriers))
            amp[user] = np.abs(response)
            phase[user] = np.angle(response)
        return {
            "csi_amp_mean": amp.mean(axis=1),
            "csi_amp_std": amp.std(axis=1),
            "csi_amp_max": amp.max(axis=1),
            "csi_phase_mean": phase.mean(axis=1),
        }

    def measure(self, user_xy: np.ndarray) -> dict[str, np.ndarray]:
        wifi_rssi, los = self._measure_links(user_xy, self.ap_positions, self.cfg.wifi_freq_ghz)
        cell_rssi, _ = self._measure_links(user_xy, self.bs_positions, self.cfg.cell_freq_ghz)
        csi = self._csi_summary(len(user_xy))
        return {"wifi_rssi": wifi_rssi, "cell_rssi": cell_rssi, "los_flags": los, **csi}


def generate_rf_dataset(cfg: ProjectConfig, output_csv: str | Path) -> pd.DataFrame:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    signal_model = PhysicsSignalModel(cfg, seed=cfg.seed)
    rows: list[dict[str, float | int]] = []
    for episode in range(cfg.episodes):
        mobility = GaussMarkovMobility(cfg, seed=cfg.seed + episode)
        positions = mobility.reset()
        for step in range(cfg.steps_per_episode):
            if step > 0:
                positions = mobility.step()
            measurement = signal_model.measure(positions)
            for user_id in range(cfg.n_users):
                row: dict[str, float | int] = {
                    "episode": episode,
                    "step": step,
                    "user_id": user_id,
                    "x": float(positions[user_id, 0]),
                    "y": float(positions[user_id, 1]),
                    "csi_amp_mean": float(measurement["csi_amp_mean"][user_id]),
                    "csi_amp_std": float(measurement["csi_amp_std"][user_id]),
                    "csi_amp_max": float(measurement["csi_amp_max"][user_id]),
                    "csi_phase_mean": float(measurement["csi_phase_mean"][user_id]),
                }
                for ap in range(cfg.n_aps):
                    row[f"wifi_rssi_{ap}"] = float(measurement["wifi_rssi"][user_id, ap])
                    row[f"los_ap_{ap}"] = int(measurement["los_flags"][user_id, ap])
                for bs in range(cfg.n_base_stations):
                    row[f"cell_rssi_{bs}"] = float(measurement["cell_rssi"][user_id, bs])
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df
```

- [ ] **Step 5: Add phase 1 script**

Create `scripts/phase1_collect_data.py`:

```python
from __future__ import annotations

import argparse

from wifi_uav.config import get_config
from wifi_uav.signal_model import generate_rf_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="smoke", choices=["smoke", "medium", "full"])
    args = parser.parse_args()
    cfg = get_config(args.preset)
    output = cfg.data_dir / "rf_dataset.csv"
    df = generate_rf_dataset(cfg, output)
    print(f"Saved RF dataset: {output}")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns):,}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run mobility and signal tests**

Run:

```bash
pytest tests/test_mobility_signal.py -v
python scripts/phase1_collect_data.py --preset smoke
```

Expected: tests PASS; script prints `Saved RF dataset: data/rf_dataset.csv`.

- [ ] **Step 7: Commit**

```bash
git add src/wifi_uav/mobility.py src/wifi_uav/signal_model.py scripts/phase1_collect_data.py tests/test_mobility_signal.py
git commit -m "feat: generate RF mobility dataset"
```

---

### Task 3: Feature Extraction

**Files:**
- Create: `src/wifi_uav/features.py`
- Create: `scripts/phase2_build_features.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Write feature extraction tests**

Create `tests/test_features.py`:

```python
import numpy as np

from wifi_uav.config import get_config
from wifi_uav.features import build_feature_splits
from wifi_uav.signal_model import generate_rf_dataset


def test_build_feature_splits_shapes_and_files(tmp_path):
    cfg = get_config("smoke")
    csv_path = tmp_path / "rf_dataset.csv"
    out_dir = tmp_path / "features"
    generate_rf_dataset(cfg, csv_path)
    splits = build_feature_splits(cfg, csv_path, out_dir)
    assert splits["X_train"].ndim == 3
    assert splits["X_train"].shape[1] == cfg.seq_len
    assert splits["y_train"].shape[1] == 2
    assert np.isfinite(splits["X_train"]).all()
    assert np.isfinite(splits["y_train"]).all()
    assert (out_dir / "X_train.npy").exists()
    assert (out_dir / "normalization.npz").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_features.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `wifi_uav.features`.

- [ ] **Step 3: Implement feature extraction**

Create `src/wifi_uav/features.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wifi_uav.config import ProjectConfig


def _user_features(group: pd.DataFrame, cfg: ProjectConfig) -> tuple[np.ndarray, np.ndarray]:
    group = group.sort_values("step")
    wifi_cols = [f"wifi_rssi_{i}" for i in range(cfg.n_aps)]
    cell_cols = [f"cell_rssi_{i}" for i in range(cfg.n_base_stations)]
    csi_cols = ["csi_amp_mean", "csi_amp_std", "csi_amp_max", "csi_phase_mean"]
    wifi = group[wifi_cols].to_numpy(dtype=np.float32)
    delta = np.vstack([np.zeros((1, cfg.n_aps), dtype=np.float32), np.diff(wifi, axis=0)])
    rolling = group[wifi_cols].rolling(window=3, min_periods=1).var().fillna(0.0).to_numpy(dtype=np.float32)
    cell = group[cell_cols].to_numpy(dtype=np.float32)
    csi = group[csi_cols].to_numpy(dtype=np.float32)
    features = np.concatenate([wifi, delta, rolling, cell, csi], axis=1)
    targets = group[["x", "y"]].to_numpy(dtype=np.float32)
    return features, targets


def build_sequences(df: pd.DataFrame, cfg: ProjectConfig) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for (_, _), group in df.groupby(["episode", "user_id"], sort=True):
        features, targets = _user_features(group, cfg)
        max_start = len(group) - cfg.seq_len - cfg.pred_horizon + 1
        for start in range(max(0, max_start)):
            end = start + cfg.seq_len
            target_idx = end + cfg.pred_horizon - 1
            xs.append(features[start:end])
            ys.append(targets[target_idx])
    if not xs:
        raise ValueError("No sequences were created; increase steps_per_episode or reduce seq_len/pred_horizon")
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def build_feature_splits(cfg: ProjectConfig, csv_path: str | Path, output_dir: str | Path) -> dict[str, np.ndarray]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    x, y = build_sequences(df, cfg)
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(x))
    x = x[order]
    y = y[order] / cfg.area_size
    n_train = int(0.70 * len(x))
    n_val = int(0.15 * len(x))
    train_x = x[:n_train]
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x = (x - mean) / std
    splits = {
        "X_train": x[:n_train],
        "y_train": y[:n_train],
        "X_val": x[n_train : n_train + n_val],
        "y_val": y[n_train : n_train + n_val],
        "X_test": x[n_train + n_val :],
        "y_test": y[n_train + n_val :],
    }
    for name, arr in splits.items():
        np.save(output_dir / f"{name}.npy", arr.astype(np.float32))
    np.savez(output_dir / "normalization.npz", mean=mean.astype(np.float32), std=std.astype(np.float32), area_size=cfg.area_size)
    return splits
```

- [ ] **Step 4: Add phase 2 script**

Create `scripts/phase2_build_features.py`:

```python
from __future__ import annotations

import argparse

from wifi_uav.config import get_config
from wifi_uav.features import build_feature_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="smoke", choices=["smoke", "medium", "full"])
    args = parser.parse_args()
    cfg = get_config(args.preset)
    splits = build_feature_splits(cfg, cfg.data_dir / "rf_dataset.csv", cfg.data_dir)
    print(f"X_train: {splits['X_train'].shape}")
    print(f"y_train: {splits['y_train'].shape}")
    print("Feature splits saved.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run feature tests and script**

Run:

```bash
pytest tests/test_features.py -v
python scripts/phase1_collect_data.py --preset smoke
python scripts/phase2_build_features.py --preset smoke
```

Expected: tests PASS; script prints `Feature splits saved.`

- [ ] **Step 6: Commit**

```bash
git add src/wifi_uav/features.py scripts/phase2_build_features.py tests/test_features.py
git commit -m "feat: build RF feature splits"
```

---

### Task 4: LSTM-Transformer Predictor

**Files:**
- Create: `src/wifi_uav/predictor.py`
- Create: `scripts/phase3_train_predictor.py`
- Create: `tests/test_predictor.py`

- [ ] **Step 1: Write predictor tests**

Create `tests/test_predictor.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_predictor.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `wifi_uav.predictor`.

- [ ] **Step 3: Implement predictor**

Create `src/wifi_uav/predictor.py` with:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMTransformerPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, n_heads: int = 4, n_tf_layers: int = 2, n_lstm_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.lstm = nn.LSTM(hidden, hidden, num_layers=n_lstm_layers, batch_first=True, dropout=dropout)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=n_heads, dim_feedforward=512, dropout=dropout, batch_first=True, norm_first=True)
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
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


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
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    loss_fn = nn.HuberLoss()
    train_loader = _loader(x_train, y_train, batch_size, True)
    val_loader = _loader(x_val, y_val, batch_size, False)
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    patience = 15
    stale = 0
    for _ in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
        sched.step()
        train_loss = total / max(1, count)
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": best_state, "history": history}, checkpoint_path)
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, model


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


@torch.no_grad()
def rmse_meters(model: nn.Module, x: np.ndarray, y: np.ndarray, area_size: float, device: str | torch.device = "cpu") -> float:
    device = torch.device(device)
    model = model.to(device)
    pred = model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
    err = (pred - y) * area_size
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
```

- [ ] **Step 4: Add phase 3 script**

Create `scripts/phase3_train_predictor.py`:

```python
from __future__ import annotations

import argparse

import numpy as np
import torch

from wifi_uav.config import get_config
from wifi_uav.predictor import rmse_meters, train_predictor_on_arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="smoke", choices=["smoke", "medium", "full"])
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    cfg = get_config(args.preset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train = np.load(cfg.data_dir / "X_train.npy")
    y_train = np.load(cfg.data_dir / "y_train.npy")
    x_val = np.load(cfg.data_dir / "X_val.npy")
    y_val = np.load(cfg.data_dir / "y_val.npy")
    x_test = np.load(cfg.data_dir / "X_test.npy")
    y_test = np.load(cfg.data_dir / "y_test.npy")
    history, model = train_predictor_on_arrays(
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=args.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.learning_rate,
        device=device,
        checkpoint_path=cfg.checkpoint_dir / "predictor_best.pt",
    )
    rmse = rmse_meters(model, x_test, y_test, cfg.area_size, device=device)
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    (cfg.result_dir / "predictor_summary.txt").write_text(f"RMSE_meters={rmse:.4f}\\n", encoding="utf-8")
    print(f"Train losses: {history['train_loss']}")
    print(f"Mean RMSE: {rmse:.2f} m")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run predictor tests and smoke training**

Run:

```bash
pytest tests/test_predictor.py -v
python scripts/phase1_collect_data.py --preset smoke
python scripts/phase2_build_features.py --preset smoke
python scripts/phase3_train_predictor.py --preset smoke --epochs 2
```

Expected: tests PASS; script prints `Mean RMSE:`.

- [ ] **Step 6: Commit**

```bash
git add src/wifi_uav/predictor.py scripts/phase3_train_predictor.py tests/test_predictor.py
git commit -m "feat: train LSTM transformer predictor"
```

---

### Task 5: Multi-UAV Coverage Environment

**Files:**
- Create: `src/wifi_uav/uav_env.py`
- Create: `scripts/phase4_sanity_env.py`
- Create: `tests/test_uav_env.py`

- [ ] **Step 1: Write environment tests**

Create `tests/test_uav_env.py`:

```python
import numpy as np

from wifi_uav.config import get_config
from wifi_uav.uav_env import UAVCoverageEnv


def test_env_reset_and_step_shapes():
    cfg = get_config("smoke")
    env = UAVCoverageEnv(cfg, seed=cfg.seed)
    obs = env.reset()
    assert len(obs) == cfg.n_uavs
    assert obs[0].ndim == 1
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_uav_env.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `wifi_uav.uav_env`.

- [ ] **Step 3: Implement UAV environment**

Create `src/wifi_uav/uav_env.py`:

```python
from __future__ import annotations

import numpy as np

from wifi_uav.config import ProjectConfig
from wifi_uav.mobility import GaussMarkovMobility


class UAVCoverageEnv:
    def __init__(self, cfg: ProjectConfig, seed: int | None = None, coverage_radius: float = 120.0):
        self.cfg = cfg
        self.coverage_radius = coverage_radius
        self.rng = np.random.default_rng(seed)
        self.mobility = GaussMarkovMobility(cfg, seed=seed)
        self.step_count = 0
        self.prev_coverage = 0.0
        self.uav_positions = np.zeros((cfg.n_uavs, 3), dtype=np.float32)
        self.user_positions = np.zeros((cfg.n_users, 2), dtype=np.float32)
        self.battery = np.ones(cfg.n_uavs, dtype=np.float32)

    @property
    def obs_dim(self) -> int:
        return 3 + (self.cfg.n_uavs - 1) * 3 + 2 + 3 + 1 + 1

    @property
    def act_dim(self) -> int:
        return 3

    def reset(self) -> list[np.ndarray]:
        self.step_count = 0
        self.user_positions = self.mobility.reset()
        xy = self.rng.uniform(0.0, self.cfg.area_size, size=(self.cfg.n_uavs, 2))
        alt = self.rng.uniform(self.cfg.uav_alt_min, self.cfg.uav_alt_max, size=(self.cfg.n_uavs, 1))
        self.uav_positions = np.concatenate([xy, alt], axis=1).astype(np.float32)
        self.battery = np.ones(self.cfg.n_uavs, dtype=np.float32)
        self.prev_coverage = self._compute_coverage()
        return self._observations()

    def step(self, actions: list[np.ndarray]) -> tuple[list[np.ndarray], float, bool, dict[str, float]]:
        actions_arr = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        displacement = actions_arr * self.cfg.v_max
        self.uav_positions += displacement
        self.uav_positions[:, 0:2] = np.clip(self.uav_positions[:, 0:2], 0.0, self.cfg.area_size)
        self.uav_positions[:, 2] = np.clip(self.uav_positions[:, 2], self.cfg.uav_alt_min, self.cfg.uav_alt_max)
        energy = np.linalg.norm(displacement, axis=1)
        self.battery = np.maximum(0.0, self.battery - 0.001 * energy)
        self.user_positions = self.mobility.step()
        coverage = self._compute_coverage()
        improvement = coverage - self.prev_coverage
        overlap = self._overlap_count()
        energy_penalty = float(energy.sum() / (self.cfg.n_uavs * self.cfg.v_max * np.sqrt(3.0)))
        reward = float(coverage + 0.5 * improvement - 0.3 * energy_penalty - 0.5 * overlap / self.cfg.n_uavs)
        self.prev_coverage = coverage
        self.step_count += 1
        done = self.step_count >= self.cfg.drl_episode_len or bool(np.all(self.battery <= 0.0))
        return self._observations(), reward, done, {"coverage": coverage, "energy": energy_penalty, "overlap": float(overlap)}

    def _compute_coverage(self) -> float:
        diff = self.user_positions[:, None, :] - self.uav_positions[None, :, :2]
        dist = np.linalg.norm(diff, axis=2)
        covered = (dist <= self.coverage_radius).any(axis=1)
        return float(covered.mean())

    def _overlap_count(self) -> int:
        count = 0
        for i in range(self.cfg.n_uavs):
            for j in range(i + 1, self.cfg.n_uavs):
                if np.linalg.norm(self.uav_positions[i, :2] - self.uav_positions[j, :2]) < 50.0:
                    count += 1
        return count

    def _observations(self) -> list[np.ndarray]:
        centroid = self.user_positions.mean(axis=0) / self.cfg.area_size
        obs = []
        for i in range(self.cfg.n_uavs):
            own = self.uav_positions[i] / np.array([self.cfg.area_size, self.cfg.area_size, self.cfg.uav_alt_max], dtype=np.float32)
            others = np.delete(self.uav_positions, i, axis=0).reshape(-1)
            others = others / np.tile(np.array([self.cfg.area_size, self.cfg.area_size, self.cfg.uav_alt_max], dtype=np.float32), self.cfg.n_uavs - 1)
            dists = np.linalg.norm(self.user_positions - self.uav_positions[i, :2], axis=1)
            top3 = np.sort(dists)[:3] / self.cfg.area_size
            item = np.concatenate([own, others, centroid, top3, [self.prev_coverage], [self.battery[i]]]).astype(np.float32)
            obs.append(item)
        return obs
```

- [ ] **Step 4: Add phase 4 script**

Create `scripts/phase4_sanity_env.py`:

```python
from __future__ import annotations

import argparse

import numpy as np

from wifi_uav.config import get_config
from wifi_uav.uav_env import UAVCoverageEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="smoke", choices=["smoke", "medium", "full"])
    args = parser.parse_args()
    cfg = get_config(args.preset)
    env = UAVCoverageEnv(cfg, seed=cfg.seed)
    obs = env.reset()
    print(f"Observation dim: {len(obs[0])}")
    print(f"Number of agents: {cfg.n_uavs}")
    for step in range(3):
        actions = [np.random.uniform(-1.0, 1.0, 3).astype(np.float32) for _ in range(cfg.n_uavs)]
        _, reward, done, info = env.step(actions)
        print(f"Step {step}: coverage={info['coverage']:.1%}, reward={reward:.3f}, done={done}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run environment tests and sanity script**

Run:

```bash
pytest tests/test_uav_env.py -v
python scripts/phase4_sanity_env.py --preset smoke
```

Expected: tests PASS; script prints 3 coverage lines.

- [ ] **Step 6: Commit**

```bash
git add src/wifi_uav/uav_env.py scripts/phase4_sanity_env.py tests/test_uav_env.py
git commit -m "feat: add multi UAV coverage environment"
```

---

### Task 6: MADDPG Training

**Files:**
- Create: `src/wifi_uav/maddpg.py`
- Create: `scripts/phase5_train_maddpg.py`
- Create: `tests/test_maddpg.py`

- [ ] **Step 1: Write MADDPG tests**

Create `tests/test_maddpg.py`:

```python
import numpy as np
import torch

from wifi_uav.maddpg import ActorNet, CriticNet, ReplayBuffer


def test_replay_buffer_samples_expected_shapes():
    buffer = ReplayBuffer(capacity=10, n_agents=3, obs_dim=17, act_dim=3, seed=1)
    obs = [np.zeros(17, dtype=np.float32) for _ in range(3)]
    acts = [np.zeros(3, dtype=np.float32) for _ in range(3)]
    for _ in range(5):
        buffer.push(obs, acts, 1.0, obs, False)
    batch = buffer.sample(4)
    assert batch["obs"].shape == (4, 3, 17)
    assert batch["actions"].shape == (4, 3, 3)
    assert batch["rewards"].shape == (4, 1)


def test_actor_and_critic_forward_shapes():
    actor = ActorNet(obs_dim=17, act_dim=3)
    critic = CriticNet(total_obs_dim=51, total_act_dim=9)
    action = actor(torch.randn(2, 17))
    q = critic(torch.randn(2, 51), torch.randn(2, 9))
    assert action.shape == (2, 3)
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)
    assert q.shape == (2, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_maddpg.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `wifi_uav.maddpg`.

- [ ] **Step 3: Implement MADDPG core**

Create `src/wifi_uav/maddpg.py`:

```python
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class ReplayBuffer:
    def __init__(self, capacity: int, n_agents: int, obs_dim: int, act_dim: int, seed: int = 42):
        self.items: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque(maxlen=capacity)
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def push(self, obs: list[np.ndarray], actions: list[np.ndarray], reward: float, next_obs: list[np.ndarray], done: bool) -> None:
        self.items.append((np.asarray(obs, dtype=np.float32), np.asarray(actions, dtype=np.float32), float(reward), np.asarray(next_obs, dtype=np.float32), bool(done)))

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = self.rng.choice(len(self.items), size=batch_size, replace=False)
        obs, actions, rewards, next_obs, dones = zip(*(self.items[i] for i in idx))
        return {
            "obs": torch.tensor(np.stack(obs), dtype=torch.float32),
            "actions": torch.tensor(np.stack(actions), dtype=torch.float32),
            "rewards": torch.tensor(np.asarray(rewards)[:, None], dtype=torch.float32),
            "next_obs": torch.tensor(np.stack(next_obs), dtype=torch.float32),
            "dones": torch.tensor(np.asarray(dones, dtype=np.float32)[:, None], dtype=torch.float32),
        }


class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, act_dim), nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CriticNet(nn.Module):
    def __init__(self, total_obs_dim: int, total_act_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(total_obs_dim + total_act_dim, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, all_obs: torch.Tensor, all_actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([all_obs, all_actions], dim=1))


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


@dataclass
class TrainingStats:
    rewards: list[float]
    coverages: list[float]
```

- [ ] **Step 4: Add smoke MADDPG training script**

Create `scripts/phase5_train_maddpg.py`:

```python
from __future__ import annotations

import argparse

import numpy as np
import torch

from wifi_uav.config import get_config
from wifi_uav.maddpg import ActorNet, ReplayBuffer
from wifi_uav.uav_env import UAVCoverageEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="smoke", choices=["smoke", "medium", "full"])
    args = parser.parse_args()
    cfg = get_config(args.preset)
    env = UAVCoverageEnv(cfg, seed=cfg.seed)
    obs = env.reset()
    obs_dim = len(obs[0])
    actors = [ActorNet(obs_dim, env.act_dim) for _ in range(cfg.n_uavs)]
    buffer = ReplayBuffer(100_000, cfg.n_uavs, obs_dim, env.act_dim, seed=cfg.seed)
    rewards: list[float] = []
    coverages: list[float] = []
    for ep in range(cfg.drl_episodes):
        obs = env.reset()
        ep_reward = 0.0
        last_cov = 0.0
        for _ in range(cfg.drl_episode_len):
            actions = []
            for i, actor in enumerate(actors):
                with torch.no_grad():
                    action = actor(torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                action = np.clip(action + np.random.normal(0.0, 0.3, size=env.act_dim), -1.0, 1.0).astype(np.float32)
                actions.append(action)
            next_obs, reward, done, info = env.step(actions)
            buffer.push(obs, actions, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            last_cov = info["coverage"]
            if done:
                break
        rewards.append(ep_reward)
        coverages.append(last_cov)
        print(f"Episode {ep + 1}: reward={ep_reward:.3f}, coverage={last_cov:.1%}, replay={len(buffer)}")
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.result_dir / "maddpg_rewards.npy", np.asarray(rewards, dtype=np.float32))
    np.save(cfg.result_dir / "maddpg_coverages.npy", np.asarray(coverages, dtype=np.float32))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run MADDPG tests and smoke script**

Run:

```bash
pytest tests/test_maddpg.py -v
python scripts/phase5_train_maddpg.py --preset smoke
```

Expected: tests PASS; script prints one line per smoke episode.

- [ ] **Step 6: Upgrade training from smoke action collection to full MADDPG updates**

Modify `src/wifi_uav/maddpg.py` after `TrainingStats` to add `MADDPGAgent` and `update_agents`:

```python
class MADDPGAgent:
    def __init__(self, obs_dim: int, act_dim: int, total_obs_dim: int, total_act_dim: int, lr: float = 1e-3, device: str = "cpu"):
        self.device = torch.device(device)
        self.actor = ActorNet(obs_dim, act_dim).to(self.device)
        self.actor_target = ActorNet(obs_dim, act_dim).to(self.device)
        self.critic = CriticNet(total_obs_dim, total_act_dim).to(self.device)
        self.critic_target = CriticNet(total_obs_dim, total_act_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=lr)

    def select_action(self, obs: np.ndarray, noise: float = 0.0) -> np.ndarray:
        with torch.no_grad():
            action = self.actor(torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)).squeeze(0).cpu().numpy()
        if noise > 0.0:
            action = action + np.random.normal(0.0, noise, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def update_agents(agents: list[MADDPGAgent], batch: dict[str, torch.Tensor], gamma: float = 0.95, tau: float = 0.01) -> None:
    device = agents[0].device
    obs = batch["obs"].to(device)
    actions = batch["actions"].to(device)
    rewards = batch["rewards"].to(device)
    next_obs = batch["next_obs"].to(device)
    dones = batch["dones"].to(device)
    bsz, n_agents, obs_dim = obs.shape
    act_dim = actions.shape[-1]
    flat_obs = obs.reshape(bsz, n_agents * obs_dim)
    flat_actions = actions.reshape(bsz, n_agents * act_dim)
    with torch.no_grad():
        next_actions = torch.stack([agent.actor_target(next_obs[:, i, :]) for i, agent in enumerate(agents)], dim=1)
        flat_next_obs = next_obs.reshape(bsz, n_agents * obs_dim)
        flat_next_actions = next_actions.reshape(bsz, n_agents * act_dim)
    for i, agent in enumerate(agents):
        with torch.no_grad():
            target_q = rewards + gamma * (1.0 - dones) * agent.critic_target(flat_next_obs, flat_next_actions)
        q = agent.critic(flat_obs, flat_actions)
        critic_loss = nn.functional.mse_loss(q, target_q)
        agent.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
        agent.critic_opt.step()
        current_actions = actions.clone()
        current_actions[:, i, :] = agent.actor(obs[:, i, :])
        actor_loss = -agent.critic(flat_obs, current_actions.reshape(bsz, n_agents * act_dim)).mean()
        agent.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
        agent.actor_opt.step()
        soft_update(agent.actor_target, agent.actor, tau)
        soft_update(agent.critic_target, agent.critic, tau)
```

- [ ] **Step 7: Modify phase 5 script to use MADDPGAgent**

Update `scripts/phase5_train_maddpg.py` imports and actor creation:

```python
from wifi_uav.maddpg import MADDPGAgent, ReplayBuffer, update_agents
```

Replace `actors = ...` with:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
agents = [
    MADDPGAgent(obs_dim, env.act_dim, obs_dim * cfg.n_uavs, env.act_dim * cfg.n_uavs, lr=cfg.learning_rate, device=device)
    for _ in range(cfg.n_uavs)
]
```

Replace action selection loop with:

```python
noise = max(0.05, 0.3 * (1.0 - ep / max(1, cfg.drl_episodes - 1)))
actions = [agent.select_action(obs[i], noise=noise) for i, agent in enumerate(agents)]
```

After `buffer.push(...)`, add:

```python
if len(buffer) >= min(cfg.batch_size, 64):
    batch = buffer.sample(min(cfg.batch_size, 64))
    update_agents(agents, batch)
```

At the end, save actors:

```python
cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
for i, agent in enumerate(agents):
    torch.save(agent.actor.state_dict(), cfg.checkpoint_dir / f"maddpg_actor_{i}.pt")
```

- [ ] **Step 8: Run MADDPG tests and training again**

Run:

```bash
pytest tests/test_maddpg.py -v
python scripts/phase5_train_maddpg.py --preset smoke
```

Expected: tests PASS; smoke training completes and writes `results/maddpg_rewards.npy`.

- [ ] **Step 9: Commit**

```bash
git add src/wifi_uav/maddpg.py scripts/phase5_train_maddpg.py tests/test_maddpg.py
git commit -m "feat: add MADDPG training loop"
```

---

### Task 7: Evaluation and Plots

**Files:**
- Create: `src/wifi_uav/evaluate.py`
- Create: `scripts/phase6_evaluate.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write evaluation tests**

Create `tests/test_evaluate.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_evaluate.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `wifi_uav.evaluate`.

- [ ] **Step 3: Implement evaluation utilities**

Create `src/wifi_uav/evaluate.py`:

```python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from wifi_uav.uav_env import UAVCoverageEnv


def moving_average(values: np.ndarray, window: int = 20) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) < window:
        return np.asarray([values[: i + 1].mean() for i in range(len(values))], dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    prefix = np.asarray([values[: i + 1].mean() for i in range(window - 1)], dtype=np.float32)
    return np.concatenate([prefix, np.convolve(values, kernel, mode="valid")])


def run_baseline_policy(env: UAVCoverageEnv, policy: str, episodes: int = 10) -> float:
    coverages: list[float] = []
    for _ in range(episodes):
        obs = env.reset()
        last_cov = 0.0
        for _ in range(env.cfg.drl_episode_len):
            if policy == "static":
                actions = [np.zeros(env.act_dim, dtype=np.float32) for _ in range(env.cfg.n_uavs)]
            elif policy == "random":
                actions = [np.random.uniform(-1.0, 1.0, env.act_dim).astype(np.float32) for _ in range(env.cfg.n_uavs)]
            elif policy == "greedy":
                centroid = env.user_positions.mean(axis=0)
                actions = []
                for pos in env.uav_positions:
                    direction = centroid - pos[:2]
                    norm = np.linalg.norm(direction)
                    xy = direction / norm if norm > 1e-6 else np.zeros(2)
                    actions.append(np.array([xy[0], xy[1], 0.0], dtype=np.float32))
            else:
                raise ValueError(f"Unknown baseline policy: {policy}")
            obs, _, done, info = env.step(actions)
            last_cov = info["coverage"]
            if done:
                break
        coverages.append(last_cov)
    return float(np.mean(coverages))


def plot_training_curves(rewards: np.ndarray, coverages: np.ndarray, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(rewards, alpha=0.35, label="raw")
    axes[0].plot(moving_average(rewards), label="moving avg")
    axes[0].set_title("Episode Reward")
    axes[0].legend()
    axes[1].plot(coverages, alpha=0.35, label="raw")
    axes[1].plot(moving_average(coverages), label="moving avg")
    axes[1].set_title("User Coverage")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_baselines(scores: dict[str, float], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(scores)
    vals = [scores[name] * 100.0 for name in names]
    sns.barplot(x=names, y=vals, ax=ax)
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Policy Coverage Comparison")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_summary(path: str | Path, predictor_rmse: float, maddpg_coverage: float, greedy_coverage: float, static_coverage: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "=== RESULTS SUMMARY ===\\n"
        f"Predictor RMSE : {predictor_rmse:.2f} m\\n"
        f"MADDPG coverage: {maddpg_coverage:.1%}\\n"
        f"vs Greedy      : {(maddpg_coverage - greedy_coverage) * 100:.1f}%\\n"
        f"vs Static hover: {(maddpg_coverage - static_coverage) * 100:.1f}%\\n"
    )
    path.write_text(text, encoding="utf-8")
```

- [ ] **Step 4: Add phase 6 script**

Create `scripts/phase6_evaluate.py`:

```python
from __future__ import annotations

import argparse
import re

import numpy as np

from wifi_uav.config import get_config
from wifi_uav.evaluate import plot_baselines, plot_training_curves, run_baseline_policy, write_summary
from wifi_uav.uav_env import UAVCoverageEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="smoke", choices=["smoke", "medium", "full"])
    args = parser.parse_args()
    cfg = get_config(args.preset)
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    rewards = np.load(cfg.result_dir / "maddpg_rewards.npy") if (cfg.result_dir / "maddpg_rewards.npy").exists() else np.zeros(1, dtype=np.float32)
    coverages = np.load(cfg.result_dir / "maddpg_coverages.npy") if (cfg.result_dir / "maddpg_coverages.npy").exists() else np.zeros(1, dtype=np.float32)
    plot_training_curves(rewards, coverages, cfg.result_dir / "training_curves.png")
    env = UAVCoverageEnv(cfg, seed=cfg.seed)
    scores = {
        "MADDPG": float(coverages[-1]) if len(coverages) else 0.0,
        "Greedy": run_baseline_policy(env, "greedy", episodes=2 if args.preset == "smoke" else 50),
        "Random": run_baseline_policy(env, "random", episodes=2 if args.preset == "smoke" else 50),
        "Static": run_baseline_policy(env, "static", episodes=2 if args.preset == "smoke" else 50),
    }
    plot_baselines(scores, cfg.result_dir / "baseline_comparison.png")
    summary_path = cfg.result_dir / "predictor_summary.txt"
    rmse = 0.0
    if summary_path.exists():
        match = re.search(r"RMSE_meters=([0-9.]+)", summary_path.read_text(encoding="utf-8"))
        rmse = float(match.group(1)) if match else 0.0
    write_summary(cfg.result_dir / "final_summary.txt", rmse, scores["MADDPG"], scores["Greedy"], scores["Static"])
    print((cfg.result_dir / "final_summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run evaluation tests and phase script**

Run:

```bash
pytest tests/test_evaluate.py -v
python scripts/phase5_train_maddpg.py --preset smoke
python scripts/phase6_evaluate.py --preset smoke
```

Expected: tests PASS; `results/training_curves.png`, `results/baseline_comparison.png`, and `results/final_summary.txt` exist.

- [ ] **Step 6: Commit**

```bash
git add src/wifi_uav/evaluate.py scripts/phase6_evaluate.py tests/test_evaluate.py
git commit -m "feat: add evaluation outputs"
```

---

### Task 8: Colab Notebook and Reports

**Files:**
- Create: `notebooks/WiFi_UAV_Full_Project_Colab.ipynb`
- Create: `reports/project_report_outline.md`
- Create: `reports/presentation_outline.md`
- Create: `scripts/create_colab_notebook.py`

- [ ] **Step 1: Create notebook generator**

Create `scripts/create_colab_notebook.py`:

```python
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def code_cell(source: str):
    return nbf.v4.new_code_cell(source.strip())


def md_cell(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md_cell("# Wi-Fi/Cellular UAV Mobility Prediction and Trajectory Optimization\\nFull Colab execution notebook."),
        code_cell("!pip install -q numpy pandas scikit-learn torch matplotlib seaborn tqdm gymnasium nbformat"),
        code_cell("import torch\\nprint('CUDA available:', torch.cuda.is_available())\\nprint('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"),
        md_cell("## Phase 0: Clone or upload project\\nUpload this repository folder to Colab or mount Google Drive before running the phase scripts."),
        code_cell("import os, sys\\nPROJECT_ROOT = '/content/wifi_uav_project'\\nif os.path.isdir(PROJECT_ROOT):\\n    os.chdir(PROJECT_ROOT)\\n    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))\\nprint(os.getcwd())"),
        md_cell("## Phase 1: RF dataset generation"),
        code_cell("!python scripts/phase1_collect_data.py --preset smoke"),
        md_cell("## Phase 2: Feature extraction"),
        code_cell("!python scripts/phase2_build_features.py --preset smoke"),
        md_cell("## Phase 3: LSTM-Transformer training"),
        code_cell("!python scripts/phase3_train_predictor.py --preset smoke --epochs 2"),
        md_cell("## Phase 4: UAV environment sanity check"),
        code_cell("!python scripts/phase4_sanity_env.py --preset smoke"),
        md_cell("## Phase 5: MADDPG training"),
        code_cell("!python scripts/phase5_train_maddpg.py --preset smoke"),
        md_cell("## Phase 6: Evaluation"),
        code_cell("!python scripts/phase6_evaluate.py --preset smoke"),
        code_cell("!zip -r wifi_uav_results.zip data models checkpoints results reports notebooks src scripts README.md pyproject.toml"),
    ]
    out = Path("notebooks/WiFi_UAV_Full_Project_Colab.ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate notebook**

Run:

```bash
python scripts/create_colab_notebook.py
```

Expected: prints `Wrote notebooks/WiFi_UAV_Full_Project_Colab.ipynb`.

- [ ] **Step 3: Add report outline**

Create `reports/project_report_outline.md`:

```markdown
# Project Report Outline

## Title
AI-Driven User Mobility Prediction and Intelligent UAV Trajectory Optimization Using Wi-Fi/Cellular Signals

## Abstract
Summarize RF data generation, LSTM-Transformer mobility prediction, MADDPG UAV coordination, and baseline evaluation.

## Problem Statement
Mobile users change position over time, causing wireless coverage demand to shift. The project predicts near-future user locations from RF signals and optimizes UAV placement for improved coverage.

## Methodology
1. RF signal simulation with path loss, shadowing, fading, RSSI, and CSI summaries
2. Feature extraction using RSSI deltas and rolling variance
3. LSTM-Transformer future position prediction
4. Gym-style UAV coverage environment
5. MADDPG cooperative control
6. Baseline comparison

## Results
Insert predictor RMSE, MADDPG coverage, baseline chart, training curves, and coverage heatmap.

## Conclusion
Discuss achieved coverage, prediction accuracy, limitations, and future work with real Sionna RT or real RF data.
```

- [ ] **Step 4: Add presentation outline**

Create `reports/presentation_outline.md`:

```markdown
# Presentation Outline

1. Title and team details
2. Motivation: dynamic users and UAV-assisted wireless coverage
3. System architecture diagram
4. RF data generation: Wi-Fi APs, cellular base stations, RSSI, CSI
5. Mobility prediction: LSTM-Transformer
6. UAV environment and reward design
7. MADDPG training approach
8. Results: RMSE, coverage, baselines
9. Limitations and future work
10. Conclusion
```

- [ ] **Step 5: Validate notebook file**

Run:

```bash
python - <<'PY'
import nbformat
nb = nbformat.read("notebooks/WiFi_UAV_Full_Project_Colab.ipynb", as_version=4)
assert len(nb.cells) >= 10
print("Notebook cells:", len(nb.cells))
PY
```

Expected: prints `Notebook cells:` with a number at least 10.

- [ ] **Step 6: Commit**

```bash
git add scripts/create_colab_notebook.py notebooks/WiFi_UAV_Full_Project_Colab.ipynb reports/project_report_outline.md reports/presentation_outline.md
git commit -m "docs: add Colab notebook and report outlines"
```

---

### Task 9: End-to-End Smoke Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run full test suite**

Run:

```bash
pytest -q
```

Expected: all tests PASS.

- [ ] **Step 2: Run full smoke pipeline**

Run:

```bash
python scripts/phase1_collect_data.py --preset smoke
python scripts/phase2_build_features.py --preset smoke
python scripts/phase3_train_predictor.py --preset smoke --epochs 2
python scripts/phase4_sanity_env.py --preset smoke
python scripts/phase5_train_maddpg.py --preset smoke
python scripts/phase6_evaluate.py --preset smoke
```

Expected:

- `data/rf_dataset.csv` exists
- `data/X_train.npy` exists
- `checkpoints/predictor_best.pt` exists
- `results/predictor_summary.txt` exists
- `results/maddpg_rewards.npy` exists
- `results/training_curves.png` exists
- `results/baseline_comparison.png` exists
- `results/final_summary.txt` exists

- [ ] **Step 3: Update README with verified smoke results**

Append this section to `README.md` with the actual smoke output values:

```markdown
## Verified Smoke Run

The local smoke pipeline verifies that every phase executes before moving to Colab full training.

Generated artifacts:

- `data/rf_dataset.csv`
- `data/X_train.npy`
- `checkpoints/predictor_best.pt`
- `results/predictor_summary.txt`
- `results/maddpg_rewards.npy`
- `results/training_curves.png`
- `results/baseline_comparison.png`
- `results/final_summary.txt`
```

- [ ] **Step 4: Commit**

```bash
git add README.md data/.gitkeep models/.gitkeep checkpoints/.gitkeep results/.gitkeep reports/.gitkeep
git commit -m "docs: record smoke verification"
```

If `.gitkeep` files do not exist, create empty files at:

```text
data/.gitkeep
models/.gitkeep
checkpoints/.gitkeep
results/.gitkeep
reports/.gitkeep
```

---

## Self-Review Checklist

- Spec coverage: Tasks 1-9 cover foundation, RF generation, feature extraction, predictor, UAV environment, MADDPG, evaluation, Colab notebook, report assets, and smoke verification.
- Placeholder scan: This plan contains no unresolved placeholder markers or intentionally vague implementation steps.
- Type consistency: `ProjectConfig`, `get_config`, `GaussMarkovMobility`, `PhysicsSignalModel`, `build_feature_splits`, `LSTMTransformerPredictor`, `UAVCoverageEnv`, `ReplayBuffer`, `ActorNet`, `CriticNet`, and evaluation utilities are named consistently across tests, scripts, and implementation steps.
