from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wifi_uav.config import ProjectConfig
from wifi_uav.evaluate import Policy
from wifi_uav.signal_model import place_access_points, place_base_stations
from wifi_uav.uav_env import UAVCoverageEnv


def _round(values: np.ndarray, digits: int = 1) -> list:
    return np.round(np.asarray(values, dtype=np.float64), digits).tolist()


def record_episode(env: UAVCoverageEnv, policy: Policy, steps: int) -> dict:
    """Roll out one episode and keep everything the web viewer needs to draw each frame.

    Frame 0 is the state right after reset; frame k is the state after the k-th action.
    """
    obs = env.reset()
    frames = {
        "uavs": [],
        "radius": [],
        "users": [],
        "observed": [],
        "covered": [],
        "coverage": [],
        "battery": [],
        "error_m": [],
    }

    def snapshot() -> None:
        frames["uavs"].append(_round(env.uav_positions))
        frames["radius"].append(_round(env.footprint_radius(env.uav_positions[:, 2])))
        frames["users"].append(_round(env.user_positions))
        frames["observed"].append(_round(env.observed_user_positions))
        frames["covered"].append(env.coverage_mask(env.user_positions).astype(int).tolist())
        frames["coverage"].append(round(float(env._compute_coverage()), 4))
        frames["battery"].append(_round(env.battery, 4))
        frames["error_m"].append(round(env.prediction_error_m(), 2))

    snapshot()
    for _ in range(steps):
        obs, _, done, _ = env.step(policy(env, obs))
        snapshot()
        if done:
            break
    return frames


def scene_metadata(cfg: ProjectConfig, env: UAVCoverageEnv, steps: int) -> dict:
    return {
        "area_size": cfg.area_size,
        "n_uavs": cfg.n_uavs,
        "n_users": cfg.n_users,
        "steps": steps,
        "dt": cfg.dt,
        "coverage_radius": cfg.coverage_radius,
        "uav_alt_min": cfg.uav_alt_min,
        "uav_alt_max": cfg.uav_alt_max,
        "optimal_altitude": round(env.optimal_altitude, 1),
        "beam_half_angle_deg": cfg.beam_half_angle_deg,
        "access_points": _round(place_access_points(cfg)),
        "base_stations": _round(place_base_stations(cfg)),
    }


def write_replays(path: str | Path, meta: dict, runs: list[dict], summary: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "summary": summary, "runs": runs}
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
