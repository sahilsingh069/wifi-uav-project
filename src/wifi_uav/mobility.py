from __future__ import annotations

import numpy as np

from wifi_uav.config import ProjectConfig


class GaussMarkovMobility:
    def __init__(self, cfg: ProjectConfig, alpha: float = 0.75, seed: int | None = None) -> None:
        self.cfg = cfg
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.positions: np.ndarray | None = None
        self.velocities: np.ndarray | None = None

    def reset(self) -> np.ndarray:
        self.positions = self.rng.uniform(0.0, self.cfg.area_size, size=(self.cfg.n_users, 2))
        angles = self.rng.uniform(0.0, 2.0 * np.pi, size=self.cfg.n_users)
        speeds = self.rng.normal(self.cfg.user_speed, self.cfg.user_speed * 0.15, size=self.cfg.n_users)
        speeds = np.clip(speeds, 0.1, self.cfg.user_speed * 2.0)
        self.velocities = np.column_stack((np.cos(angles), np.sin(angles))) * speeds[:, None]
        return self.positions.copy()

    def step(self) -> np.ndarray:
        if self.positions is None or self.velocities is None:
            return self.reset()

        random_angles = self.rng.uniform(0.0, 2.0 * np.pi, size=self.cfg.n_users)
        random_speeds = self.rng.normal(self.cfg.user_speed, self.cfg.user_speed * 0.25, size=self.cfg.n_users)
        random_speeds = np.clip(random_speeds, 0.0, self.cfg.user_speed * 2.5)
        random_velocities = np.column_stack((np.cos(random_angles), np.sin(random_angles))) * random_speeds[:, None]
        noise = self.rng.normal(0.0, self.cfg.user_speed * 0.1, size=(self.cfg.n_users, 2))

        beta = np.sqrt(max(0.0, 1.0 - self.alpha**2))
        self.velocities = self.alpha * self.velocities + beta * random_velocities + noise

        speeds = np.linalg.norm(self.velocities, axis=1)
        max_speed = max(self.cfg.user_speed * 2.5, 0.1)
        scale = np.ones_like(speeds)
        moving = speeds > 1e-9
        scale[moving] = np.clip(speeds[moving], 0.1, max_speed) / speeds[moving]
        self.velocities *= scale[:, None]

        next_positions = self.positions + self.velocities * self.cfg.dt
        for axis in range(2):
            below = next_positions[:, axis] < 0.0
            above = next_positions[:, axis] > self.cfg.area_size
            self.velocities[below | above, axis] *= -1.0
            next_positions[below, axis] = -next_positions[below, axis]
            next_positions[above, axis] = 2.0 * self.cfg.area_size - next_positions[above, axis]

        self.positions = np.clip(next_positions, 0.0, self.cfg.area_size)
        return self.positions.copy()
