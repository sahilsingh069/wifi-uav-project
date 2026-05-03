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
        if actions_arr.shape != (self.cfg.n_uavs, self.act_dim):
            raise ValueError(f"actions must have shape {(self.cfg.n_uavs, self.act_dim)}, got {actions_arr.shape}")

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
        info = {"coverage": coverage, "energy": energy_penalty, "overlap": float(overlap)}
        return self._observations(), reward, done, info

    def _compute_coverage(self) -> float:
        diff = self.user_positions[:, None, :] - self.uav_positions[None, :, :2]
        dist = np.linalg.norm(diff, axis=2)
        covered = (dist <= self.coverage_radius).any(axis=1)
        return float(covered.mean())

    def _overlap_count(self) -> int:
        count = 0
        for first in range(self.cfg.n_uavs):
            for second in range(first + 1, self.cfg.n_uavs):
                distance = np.linalg.norm(self.uav_positions[first, :2] - self.uav_positions[second, :2])
                if distance < 50.0:
                    count += 1
        return count

    def _observations(self) -> list[np.ndarray]:
        centroid = self.user_positions.mean(axis=0) / self.cfg.area_size
        normalizer = np.array([self.cfg.area_size, self.cfg.area_size, self.cfg.uav_alt_max], dtype=np.float32)
        observations = []
        for idx in range(self.cfg.n_uavs):
            own = self.uav_positions[idx] / normalizer
            others = np.delete(self.uav_positions, idx, axis=0).reshape(-1)
            others = others / np.tile(normalizer, self.cfg.n_uavs - 1)
            distances = np.linalg.norm(self.user_positions - self.uav_positions[idx, :2], axis=1)
            top3 = np.sort(distances)[:3] / self.cfg.area_size
            if len(top3) < 3:
                top3 = np.pad(top3, (0, 3 - len(top3)), constant_values=1.0)
            obs = np.concatenate([own, others, centroid, top3, [self.prev_coverage], [self.battery[idx]]])
            observations.append(obs.astype(np.float32))
        return observations
