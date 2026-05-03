from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wifi_uav.config import ProjectConfig
from wifi_uav.mobility import GaussMarkovMobility


def place_access_points(cfg: ProjectConfig) -> np.ndarray:
    base_x = np.linspace(0.2 * cfg.area_size, 0.8 * cfg.area_size, 4)
    base_y = np.linspace(0.25 * cfg.area_size, 0.75 * cfg.area_size, 2)
    positions = [(x, y) for y in base_y for x in base_x]

    if cfg.n_aps > len(positions):
        cols = int(np.ceil(np.sqrt(cfg.n_aps)))
        rows = int(np.ceil(cfg.n_aps / cols))
        grid_x = np.linspace(0.1 * cfg.area_size, 0.9 * cfg.area_size, cols)
        grid_y = np.linspace(0.1 * cfg.area_size, 0.9 * cfg.area_size, rows)
        seen = {(round(x, 9), round(y, 9)) for x, y in positions}
        for y in grid_y:
            for x in grid_x:
                key = (round(float(x), 9), round(float(y), 9))
                if key not in seen:
                    positions.append((float(x), float(y)))
                    seen.add(key)
                if len(positions) == cfg.n_aps:
                    break
            if len(positions) == cfg.n_aps:
                break

    xy = np.array(positions[: cfg.n_aps], dtype=float)
    return np.column_stack((xy, np.full(cfg.n_aps, cfg.ap_height)))


def place_base_stations(cfg: ProjectConfig) -> np.ndarray:
    positions = [
        (0.05 * cfg.area_size, 0.05 * cfg.area_size),
        (0.95 * cfg.area_size, 0.20 * cfg.area_size),
        (0.50 * cfg.area_size, 0.95 * cfg.area_size),
    ]

    if cfg.n_base_stations > len(positions):
        center = 0.5 * cfg.area_size
        radius = 0.42 * cfg.area_size
        for idx in range(cfg.n_base_stations - len(positions)):
            angle = 2.0 * np.pi * idx / (cfg.n_base_stations - len(positions))
            x = np.clip(center + radius * np.cos(angle), 0.05 * cfg.area_size, 0.95 * cfg.area_size)
            y = np.clip(center + radius * np.sin(angle), 0.05 * cfg.area_size, 0.95 * cfg.area_size)
            positions.append((float(x), float(y)))

    xy = np.array(positions[: cfg.n_base_stations], dtype=float)
    return np.column_stack((xy, np.full(cfg.n_base_stations, cfg.bs_height)))


class PhysicsSignalModel:
    def __init__(self, cfg: ProjectConfig, seed: int | None = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.ap_positions = place_access_points(cfg)
        self.bs_positions = place_base_stations(cfg)

    def _path_loss_db(self, distances: np.ndarray, freq_ghz: float, los: np.ndarray) -> np.ndarray:
        distances_m = np.maximum(distances, 1.0)
        fspl = 32.44 + 20.0 * np.log10(distances_m / 1000.0) + 20.0 * np.log10(freq_ghz * 1000.0)
        nlos_penalty = self.rng.normal(15.0, 3.0, size=distances.shape)
        return fspl + np.where(los, 0.0, nlos_penalty)

    def _los_probability(self, distances: np.ndarray) -> np.ndarray:
        distances_m = np.maximum(distances, 1.0)
        return np.clip(0.85 * np.exp(-distances_m / 250.0) + 0.10, 0.05, 0.95)

    def _measure_links(self, user_xy: np.ndarray, tx_xyz: np.ndarray, freq_ghz: float) -> tuple[np.ndarray, np.ndarray]:
        user_xyz = np.column_stack((user_xy, np.full(len(user_xy), 1.5)))
        distances = np.linalg.norm(user_xyz[:, None, :] - tx_xyz[None, :, :], axis=2)
        los_flags = self.rng.random(distances.shape) < self._los_probability(distances)
        path_loss = self._path_loss_db(distances, freq_ghz, los_flags)
        shadowing = self.rng.normal(0.0, 4.0, size=distances.shape)
        rayleigh = self.rng.rayleigh(scale=1.0, size=distances.shape)
        fading_db = 20.0 * np.log10(np.maximum(rayleigh, 1e-3))
        rssi = self.cfg.tx_power_dbm - path_loss + shadowing + fading_db
        return rssi, los_flags

    def _csi_summary(self, n_users: int) -> dict[str, np.ndarray]:
        n_subcarriers = 52
        amp = np.zeros((n_users, n_subcarriers), dtype=float)
        phase = np.zeros((n_users, n_subcarriers), dtype=float)
        subcarrier_idx = np.arange(n_subcarriers, dtype=float)

        for user_idx in range(n_users):
            n_paths = int(self.rng.integers(3, 6))
            gains = self.rng.rayleigh(scale=1.0, size=n_paths)
            delays = self.rng.uniform(0.0, 0.8, size=n_paths)
            phases = self.rng.uniform(-np.pi, np.pi, size=n_paths)
            response = np.zeros(n_subcarriers, dtype=complex)
            for gain, delay, phase_offset in zip(gains, delays, phases):
                response += gain * np.exp(1j * (phase_offset - 2.0 * np.pi * delay * subcarrier_idx / n_subcarriers))
            amp[user_idx] = np.abs(response)
            phase[user_idx] = np.angle(response)

        return {
            "csi_amp_mean": amp.mean(axis=1),
            "csi_amp_std": amp.std(axis=1),
            "csi_amp_max": amp.max(axis=1),
            "csi_phase_mean": phase.mean(axis=1),
        }

    def measure(self, user_xy: np.ndarray) -> dict[str, np.ndarray]:
        wifi_rssi, los_flags = self._measure_links(user_xy, self.ap_positions, self.cfg.wifi_freq_ghz)
        cell_rssi, _ = self._measure_links(user_xy, self.bs_positions, self.cfg.cell_freq_ghz)
        return {
            "wifi_rssi": wifi_rssi,
            "cell_rssi": cell_rssi,
            "los_flags": los_flags,
            **self._csi_summary(len(user_xy)),
        }


def generate_rf_dataset(cfg: ProjectConfig, output_csv: str | Path) -> pd.DataFrame:
    mobility = GaussMarkovMobility(cfg, seed=cfg.seed)
    signal_model = PhysicsSignalModel(cfg, seed=cfg.seed)
    rows: list[dict[str, float | int | bool]] = []

    for episode in range(cfg.episodes):
        user_xy = mobility.reset()
        for step in range(cfg.steps_per_episode):
            if step > 0:
                user_xy = mobility.step()
            sample = signal_model.measure(user_xy)
            for user_id, (x_pos, y_pos) in enumerate(user_xy):
                row: dict[str, float | int | bool] = {
                    "episode": episode,
                    "step": step,
                    "user_id": user_id,
                    "x": float(x_pos),
                    "y": float(y_pos),
                    "csi_amp_mean": float(sample["csi_amp_mean"][user_id]),
                    "csi_amp_std": float(sample["csi_amp_std"][user_id]),
                    "csi_amp_max": float(sample["csi_amp_max"][user_id]),
                    "csi_phase_mean": float(sample["csi_phase_mean"][user_id]),
                }
                for ap_idx in range(cfg.n_aps):
                    row[f"wifi_rssi_{ap_idx}"] = float(sample["wifi_rssi"][user_id, ap_idx])
                    row[f"los_ap_{ap_idx}"] = bool(sample["los_flags"][user_id, ap_idx])
                for bs_idx in range(cfg.n_base_stations):
                    row[f"cell_rssi_{bs_idx}"] = float(sample["cell_rssi"][user_id, bs_idx])
                rows.append(row)

    df = pd.DataFrame(rows)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
