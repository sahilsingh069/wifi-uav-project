from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wifi_uav.config import ProjectConfig
from wifi_uav.mobility import GaussMarkovMobility
from wifi_uav.signal_model import PhysicsSignalModel

if TYPE_CHECKING:
    from wifi_uav.predictor import OnlinePositionPredictor


class UAVCoverageEnv:
    """Cooperative multi-UAV coverage environment.

    Coverage is always scored against the true user positions, but what the UAVs *observe*
    depends on ``predictor``: without one they see ground truth (oracle mode); with one, users'
    positions are inferred from simulated Wi-Fi/cellular RF measurements by the trained
    LSTM-Transformer, which is the setting a real deployment would face.
    """

    def __init__(
        self,
        cfg: ProjectConfig,
        seed: int | None = None,
        coverage_radius: float | None = None,
        predictor: OnlinePositionPredictor | None = None,
    ):
        self.cfg = cfg
        self.coverage_radius = float(cfg.coverage_radius if coverage_radius is None else coverage_radius)
        beam = np.deg2rad(cfg.beam_half_angle_deg)
        self.tan_beam = float(np.tan(beam))
        # Link-budget range chosen so the best footprint (at altitude R*cos(beam)) equals coverage_radius.
        self.link_range = self.coverage_radius / float(np.sin(beam))
        self.rng = np.random.default_rng(seed)
        self.mobility = GaussMarkovMobility(cfg, seed=seed)
        self.predictor = predictor
        self.signal_model = PhysicsSignalModel(cfg, seed=None if seed is None else seed + 1) if predictor else None
        self.step_count = 0
        self.prev_coverage = 0.0
        self.uav_positions = np.zeros((cfg.n_uavs, 3), dtype=np.float32)
        self.user_positions = np.zeros((cfg.n_users, 2), dtype=np.float32)
        self.observed_user_positions = np.zeros((cfg.n_users, 2), dtype=np.float32)
        self.battery = np.ones(cfg.n_uavs, dtype=np.float32)

    # Number of nearest observed users each UAV sees as relative vectors.
    K_NEAREST = 6

    @property
    def obs_dim(self) -> int:
        # own (x, y, h, battery) + other UAVs relative (dx, dy, dh) + K users (dx, dy, covered_by_other)
        # + user centroid relative (dx, dy) + own share of users + team coverage
        return 4 + (self.cfg.n_uavs - 1) * 3 + self.K_NEAREST * 3 + 2 + 2

    @property
    def act_dim(self) -> int:
        return 3

    @property
    def uses_predictor(self) -> bool:
        return self.predictor is not None

    @property
    def optimal_altitude(self) -> float:
        best = self.link_range / np.sqrt(1.0 + self.tan_beam**2)
        return float(np.clip(best, self.cfg.uav_alt_min, self.cfg.uav_alt_max))

    def footprint_radius(self, altitude: np.ndarray | float) -> np.ndarray:
        """Ground coverage radius: limited by antenna beam width at low altitude and by link
        budget at high altitude, giving an optimal altitude in between."""
        h = np.asarray(altitude, dtype=np.float32)
        beam_limited = h * self.tan_beam
        range_limited = np.sqrt(np.maximum(self.link_range**2 - h**2, 0.0))
        return np.minimum(beam_limited, range_limited)

    def reset(self) -> list[np.ndarray]:
        self.step_count = 0
        self.user_positions = self.mobility.reset()
        xy = self.rng.uniform(0.0, self.cfg.area_size, size=(self.cfg.n_uavs, 2))
        alt = self.rng.uniform(self.cfg.uav_alt_min, self.cfg.uav_alt_max, size=(self.cfg.n_uavs, 1))
        self.uav_positions = np.concatenate([xy, alt], axis=1).astype(np.float32)
        self.battery = np.ones(self.cfg.n_uavs, dtype=np.float32)
        if self.predictor is not None:
            self.signal_model.reset_links()
            self.predictor.reset()
        self._sense_users()
        self.prev_coverage = self._compute_coverage()
        return self._observations()

    def step(self, actions: list[np.ndarray]) -> tuple[list[np.ndarray], float, bool, dict[str, float]]:
        actions_arr = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        if actions_arr.shape != (self.cfg.n_uavs, self.act_dim):
            raise ValueError(f"actions must have shape {(self.cfg.n_uavs, self.act_dim)}, got {actions_arr.shape}")

        # UAVs with an empty battery have landed and can no longer move.
        actions_arr[self.battery <= 0.0] = 0.0
        previous_positions = self.uav_positions.copy()
        self.uav_positions += actions_arr * self.cfg.v_max
        self.uav_positions[:, 0:2] = np.clip(self.uav_positions[:, 0:2], 0.0, self.cfg.area_size)
        self.uav_positions[:, 2] = np.clip(self.uav_positions[:, 2], self.cfg.uav_alt_min, self.cfg.uav_alt_max)
        distance_flown = float(np.linalg.norm(self.uav_positions - previous_positions, axis=1).mean())

        action_effort = np.linalg.norm(actions_arr, axis=1)
        self.battery = np.maximum(0.0, self.battery - self.cfg.battery_drain_rate * action_effort)
        self.user_positions = self.mobility.step()
        self._sense_users()

        coverage = self._compute_coverage()
        improvement = coverage - self.prev_coverage
        pair_overlap = self._pairwise_overlap()
        overlap = self._overlap_penalty(pair_overlap)
        energy_penalty = float(action_effort.sum() / (self.cfg.n_uavs * np.sqrt(3.0)))
        reward = float(
            coverage + 0.5 * improvement - 0.3 * energy_penalty - self.cfg.overlap_penalty_weight * overlap
        )
        agent_rewards = self._agent_rewards(coverage, improvement, action_effort, pair_overlap)

        self.prev_coverage = coverage
        self.step_count += 1
        done = self.step_count >= self.cfg.drl_episode_len or bool(np.all(self.battery <= 0.0))
        info = {
            "coverage": coverage,
            "energy": energy_penalty,
            "overlap": overlap,
            "prediction_error_m": self.prediction_error_m(),
            "mean_battery": float(self.battery.mean()),
            "agent_rewards": agent_rewards,
            "distance_m": distance_flown,
        }
        return self._observations(), reward, done, info

    def prediction_error_m(self) -> float:
        """Mean distance between what UAVs observe and where users really are (0 in oracle mode)."""
        return float(np.linalg.norm(self.observed_user_positions - self.user_positions, axis=1).mean())

    def coverage_mask(self, points: np.ndarray) -> np.ndarray:
        """Which ground points are inside the footprint of at least one airborne UAV."""
        active = self.battery > 0.0
        if not active.any():
            return np.zeros(len(points), dtype=bool)
        uavs = self.uav_positions[active]
        dist = np.linalg.norm(points[:, None, :] - uavs[None, :, :2], axis=2)
        return (dist <= self.footprint_radius(uavs[:, 2])[None, :]).any(axis=1)

    def _sense_users(self) -> None:
        if self.predictor is None:
            self.observed_user_positions = self.user_positions.copy()
            return
        self.predictor.observe(self.signal_model.measure(self.user_positions))
        self.observed_user_positions = self.predictor.predict()

    def _compute_coverage(self) -> float:
        return float(self.coverage_mask(self.user_positions).mean())

    def _pairwise_overlap(self) -> np.ndarray:
        """(n, n) footprint overlap in [0, 1]: 0 when discs are disjoint, 1 when stacked."""
        radii = self.footprint_radius(self.uav_positions[:, 2])
        dist = np.linalg.norm(self.uav_positions[:, None, :2] - self.uav_positions[None, :, :2], axis=2)
        reach = np.maximum(radii[:, None] + radii[None, :], 1e-6)
        overlap = np.clip(1.0 - dist / reach, 0.0, 1.0)
        np.fill_diagonal(overlap, 0.0)
        return overlap

    def _overlap_penalty(self, pair_overlap: np.ndarray | None = None) -> float:
        """Mean pairwise footprint overlap across all UAV pairs."""
        n = self.cfg.n_uavs
        if n < 2:
            return 0.0
        pair_overlap = self._pairwise_overlap() if pair_overlap is None else pair_overlap
        return float(pair_overlap.sum() / (n * (n - 1)))

    def _agent_rewards(
        self,
        coverage: float,
        improvement: float,
        action_effort: np.ndarray,
        pair_overlap: np.ndarray,
    ) -> np.ndarray:
        """Per-UAV reward: the team terms plus a difference reward.

        ``unique`` is the share of users that only this UAV covers, i.e. how much team coverage
        would drop without it. It tells each agent what *its* move contributed, which a single
        shared reward cannot.
        """
        n = self.cfg.n_uavs
        active = self.battery > 0.0
        radii = self.footprint_radius(self.uav_positions[:, 2])
        dist = np.linalg.norm(self.user_positions[:, None, :] - self.uav_positions[None, :, :2], axis=2)
        inside = (dist <= radii[None, :]) & active[None, :]
        unique = (inside & (inside.sum(axis=1, keepdims=True) == 1)).mean(axis=0)
        own_overlap = pair_overlap.sum(axis=1) / max(n - 1, 1)
        return (
            coverage
            + 0.5 * improvement
            + self.cfg.marginal_reward_weight * unique
            - 0.3 * action_effort / np.sqrt(3.0)
            - self.cfg.overlap_penalty_weight * own_overlap
        ).astype(np.float32)

    def _observations(self) -> list[np.ndarray]:
        """Egocentric observations: everything is relative to the observing UAV, so the actor can
        tell which *direction* users and teammates are in (absolute distances alone cannot)."""
        area = self.cfg.area_size
        alt_max = self.cfg.uav_alt_max
        users = self.observed_user_positions
        radii = self.footprint_radius(self.uav_positions[:, 2])
        active = self.battery > 0.0
        user_uav_dist = np.linalg.norm(users[:, None, :] - self.uav_positions[None, :, :2], axis=2)
        in_footprint = (user_uav_dist <= radii[None, :]) & active[None, :]
        observations = []
        for idx in range(self.cfg.n_uavs):
            pos = self.uav_positions[idx]
            own = [pos[0] / area, pos[1] / area, pos[2] / alt_max, self.battery[idx]]

            others = np.delete(self.uav_positions, idx, axis=0) - pos
            others = others[np.argsort(np.linalg.norm(others[:, :2], axis=1))]
            others = (others / np.array([area, area, alt_max], dtype=np.float32)).reshape(-1)

            rel_users = (users - pos[:2]) / area
            nearest = np.argsort(user_uav_dist[:, idx])[: self.K_NEAREST]
            covered_by_other = np.delete(in_footprint, idx, axis=1).any(axis=1)
            user_feats = np.column_stack([rel_users[nearest], covered_by_other[nearest].astype(np.float32)])
            if len(nearest) < self.K_NEAREST:
                pad = np.tile([[0.0, 0.0, 1.0]], (self.K_NEAREST - len(nearest), 1))
                user_feats = np.vstack([user_feats, pad])

            centroid = rel_users.mean(axis=0)
            own_share = in_footprint[:, idx].mean()
            obs = np.concatenate(
                [own, others, user_feats.reshape(-1), centroid, [own_share, self.prev_coverage]]
            )
            observations.append(obs.astype(np.float32))
        return observations
