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


# 3GPP TR 38.901 parameters: Wi-Fi APs are low street-level radios (UMi street canyon),
# cellular base stations are rooftop macro cells (UMa).
_SCENARIOS = {
    "umi": {"los_dist": 36.0, "sigma_los": 4.0, "sigma_nlos": 7.82},
    "uma": {"los_dist": 63.0, "sigma_los": 4.0, "sigma_nlos": 6.0},
}
USER_HEIGHT = 1.5
N_SUBCARRIERS = 52
MAX_PATHS = 6
RSSI_AVERAGING_SAMPLES = 8


def los_probability(d2d: np.ndarray, scenario: str) -> np.ndarray:
    """3GPP TR 38.901 Table 7.4.2-1 LoS probability (user height <= 13 m)."""
    d2d = np.maximum(d2d, 1e-6)
    los_dist = _SCENARIOS[scenario]["los_dist"]
    prob = 18.0 / d2d + np.exp(-d2d / los_dist) * (1.0 - 18.0 / d2d)
    return np.where(d2d <= 18.0, 1.0, np.clip(prob, 0.0, 1.0))


def path_loss_db(d3d: np.ndarray, freq_ghz: float, los: np.ndarray, scenario: str) -> np.ndarray:
    """3GPP TR 38.901 Table 7.4.1-1 path loss (single-slope LoS form)."""
    d3d = np.maximum(d3d, 10.0)
    log_d = np.log10(d3d)
    log_f = np.log10(freq_ghz)
    if scenario == "umi":
        pl_los = 32.4 + 21.0 * log_d + 20.0 * log_f
        pl_nlos = 22.4 + 35.3 * log_d + 21.3 * log_f - 0.3 * (USER_HEIGHT - 1.5)
    elif scenario == "uma":
        pl_los = 28.0 + 22.0 * log_d + 20.0 * log_f
        pl_nlos = 13.54 + 39.08 * log_d + 20.0 * log_f - 0.6 * (USER_HEIGHT - 1.5)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return np.where(los, pl_los, np.maximum(pl_los, pl_nlos))


class PhysicsSignalModel:
    """Stateful RF simulator.

    Shadowing and LoS state are spatially correlated along each user's trajectory
    (Gudmundson model), so consecutive RSSI samples behave like a real walk instead of
    independent draws. Call ``reset_links`` whenever users teleport (new episode).
    """

    def __init__(self, cfg: ProjectConfig, seed: int | None = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.ap_positions = place_access_points(cfg)
        self.bs_positions = place_base_stations(cfg)
        self._link_state: dict[str, dict[str, np.ndarray]] = {}

    def reset_links(self) -> None:
        self._link_state = {}

    def _measure_links(
        self,
        user_xy: np.ndarray,
        tx_xyz: np.ndarray,
        freq_ghz: float,
        scenario: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        user_xyz = np.column_stack((user_xy, np.full(len(user_xy), USER_HEIGHT)))
        d3d = np.linalg.norm(user_xyz[:, None, :] - tx_xyz[None, :, :], axis=2)
        d2d = np.linalg.norm(user_xy[:, None, :] - tx_xyz[None, :, :2], axis=2)
        p_los = los_probability(d2d, scenario)

        fresh_los = self.rng.random(d2d.shape) < p_los
        fresh_shadow = self.rng.normal(0.0, 1.0, size=d2d.shape)
        state = self._link_state.get(scenario)
        if state is None or state["xy"].shape != user_xy.shape:
            los, shadow = fresh_los, fresh_shadow
        else:
            moved = np.linalg.norm(user_xy - state["xy"], axis=1, keepdims=True)
            rho = np.exp(-moved / self.cfg.shadowing_decorrelation_m)
            keep_los = self.rng.random(d2d.shape) < rho
            los = np.where(keep_los, state["los"], fresh_los)
            shadow = rho * state["shadow"] + np.sqrt(1.0 - rho**2) * fresh_shadow
        self._link_state[scenario] = {"xy": user_xy.copy(), "los": los, "shadow": shadow}

        params = _SCENARIOS[scenario]
        sigma = np.where(los, params["sigma_los"], params["sigma_nlos"])
        # Receivers report RSSI averaged over several packets, which smooths Rayleigh fading.
        fading_power = self.rng.exponential(1.0, size=d2d.shape + (RSSI_AVERAGING_SAMPLES,)).mean(axis=-1)
        fading_db = 10.0 * np.log10(np.maximum(fading_power, 1e-6))
        rssi = self.cfg.tx_power_dbm - path_loss_db(d3d, freq_ghz, los, scenario) + sigma * shadow + fading_db
        return rssi, los, d3d

    def _csi_summary(self, serving_dist: np.ndarray, serving_los: np.ndarray) -> dict[str, np.ndarray]:
        """Multipath CSI on the serving AP link.

        LoS links get a dominant Rician component (K = 9 dB) and the delay spread grows with
        distance, so CSI statistics carry information about where the user is.
        """
        n_users = len(serving_dist)
        subcarrier_idx = np.arange(N_SUBCARRIERS, dtype=float)
        n_paths = self.rng.integers(3, MAX_PATHS + 1, size=n_users)
        active = np.arange(MAX_PATHS)[None, :] < n_paths[:, None]
        gains = self.rng.rayleigh(scale=1.0, size=(n_users, MAX_PATHS)) * active
        max_delay = np.clip(0.05 + serving_dist / 400.0, 0.05, 0.9)
        delays = self.rng.uniform(0.0, 1.0, size=(n_users, MAX_PATHS)) * max_delay[:, None]
        phases = self.rng.uniform(-np.pi, np.pi, size=(n_users, MAX_PATHS))

        k_factor = 10.0 ** (9.0 / 10.0)
        gains[:, 0] = np.where(serving_los, np.sqrt(k_factor), gains[:, 0])
        delays[:, 0] = np.where(serving_los, 0.0, delays[:, 0])

        angle = phases[:, :, None] - 2.0 * np.pi * delays[:, :, None] * subcarrier_idx[None, None, :] / N_SUBCARRIERS
        response = (gains[:, :, None] * np.exp(1j * angle)).sum(axis=1)
        amp = np.abs(response)
        phase = np.angle(response)
        return {
            "csi_amp_mean": amp.mean(axis=1),
            "csi_amp_std": amp.std(axis=1),
            "csi_amp_max": amp.max(axis=1),
            "csi_phase_mean": phase.mean(axis=1),
        }

    def measure(self, user_xy: np.ndarray) -> dict[str, np.ndarray]:
        user_xy = np.asarray(user_xy, dtype=float)
        wifi_rssi, los_flags, wifi_dist = self._measure_links(user_xy, self.ap_positions, self.cfg.wifi_freq_ghz, "umi")
        cell_rssi, _, _ = self._measure_links(user_xy, self.bs_positions, self.cfg.cell_freq_ghz, "uma")
        serving = wifi_rssi.argmax(axis=1)
        rows = np.arange(len(user_xy))
        return {
            "wifi_rssi": wifi_rssi,
            "cell_rssi": cell_rssi,
            "los_flags": los_flags,
            **self._csi_summary(wifi_dist[rows, serving], los_flags[rows, serving]),
        }


CSI_COLUMNS = ["csi_amp_mean", "csi_amp_std", "csi_amp_max", "csi_phase_mean"]


def generate_rf_dataset(cfg: ProjectConfig, output_csv: str | Path) -> pd.DataFrame:
    mobility = GaussMarkovMobility(cfg, seed=cfg.seed)
    signal_model = PhysicsSignalModel(cfg, seed=cfg.seed)
    columns: dict[str, list[np.ndarray]] = {}

    def add(name: str, values: np.ndarray) -> None:
        columns.setdefault(name, []).append(np.asarray(values))

    user_ids = np.arange(cfg.n_users)
    for episode in range(cfg.episodes):
        user_xy = mobility.reset()
        signal_model.reset_links()
        for step in range(cfg.steps_per_episode):
            if step > 0:
                user_xy = mobility.step()
            sample = signal_model.measure(user_xy)
            add("episode", np.full(cfg.n_users, episode))
            add("step", np.full(cfg.n_users, step))
            add("user_id", user_ids)
            add("x", user_xy[:, 0])
            add("y", user_xy[:, 1])
            for name in CSI_COLUMNS:
                add(name, sample[name])
            for ap_idx in range(cfg.n_aps):
                add(f"wifi_rssi_{ap_idx}", sample["wifi_rssi"][:, ap_idx])
                add(f"los_ap_{ap_idx}", sample["los_flags"][:, ap_idx])
            for bs_idx in range(cfg.n_base_stations):
                add(f"cell_rssi_{bs_idx}", sample["cell_rssi"][:, bs_idx])

    df = pd.DataFrame({name: np.concatenate(parts) for name, parts in columns.items()})
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
