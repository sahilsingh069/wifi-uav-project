from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from wifi_uav.config import ProjectConfig
from wifi_uav.predictor import OnlinePositionPredictor
from wifi_uav.uav_env import UAVCoverageEnv


def moving_average(values: np.ndarray, window: int = 20) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    if len(values) < window:
        return np.asarray([values[: idx + 1].mean() for idx in range(len(values))], dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    prefix = np.asarray([values[: idx + 1].mean() for idx in range(window - 1)], dtype=np.float32)
    return np.concatenate([prefix, np.convolve(values, kernel, mode="valid")])


Policy = Callable[[UAVCoverageEnv, list[np.ndarray]], list[np.ndarray]]


def make_env(
    cfg: ProjectConfig,
    seed: int,
    use_predictor: bool,
    device: str = "cpu",
) -> UAVCoverageEnv:
    """Build the coverage env, wiring in the trained LSTM-Transformer when requested."""
    predictor = None
    if use_predictor:
        checkpoint = cfg.checkpoint_dir / "predictor_best.pt"
        normalization = cfg.data_dir / "normalization.npz"
        missing = [str(path) for path in (checkpoint, normalization) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Predicted-position mode needs phases 2-3 outputs; missing {missing}. "
                "Run phase2/phase3 first or pass --positions oracle."
            )
        predictor = OnlinePositionPredictor.from_files(cfg, checkpoint, normalization, device=device)
    return UAVCoverageEnv(cfg, seed=seed, predictor=predictor)


def static_policy(env: UAVCoverageEnv, obs: list[np.ndarray]) -> list[np.ndarray]:
    return [np.zeros(env.act_dim, dtype=np.float32) for _ in range(env.cfg.n_uavs)]


def random_policy(env: UAVCoverageEnv, obs: list[np.ndarray]) -> list[np.ndarray]:
    return [env.rng.uniform(-1.0, 1.0, env.act_dim).astype(np.float32) for _ in range(env.cfg.n_uavs)]


def greedy_policy(env: UAVCoverageEnv, obs: list[np.ndarray], iterations: int = 5) -> list[np.ndarray]:
    """K-means on the users each UAV can observe; UAV k flies to cluster k at the best altitude.

    Clusters are seeded from the UAVs' own positions so each UAV keeps a stable assignment.
    """
    users = env.observed_user_positions
    centers = env.uav_positions[:, :2].astype(np.float64).copy()
    for _ in range(iterations):
        labels = np.linalg.norm(users[:, None, :] - centers[None, :, :], axis=2).argmin(axis=1)
        for k in range(len(centers)):
            members = users[labels == k]
            if len(members):
                centers[k] = members.mean(axis=0)
    actions = []
    for pos, target in zip(env.uav_positions, centers):
        delta = np.append(target - pos[:2], env.optimal_altitude - pos[2])
        actions.append(np.clip(delta / env.cfg.v_max, -1.0, 1.0).astype(np.float32))
    return actions


def maddpg_policy(agents: list) -> Policy:
    def act(env: UAVCoverageEnv, obs: list[np.ndarray]) -> list[np.ndarray]:
        return [agent.select_action(obs[idx], noise=0.0) for idx, agent in enumerate(agents)]

    return act


BASELINES: dict[str, Policy] = {"static": static_policy, "random": random_policy, "greedy": greedy_policy}


@dataclass
class PolicyResult:
    mean_coverage: float
    final_coverage: float
    mean_prediction_error_m: float
    heatmap: np.ndarray


def evaluate_policy(env: UAVCoverageEnv, policy: Policy, episodes: int = 10, grid: int = 50) -> PolicyResult:
    """Roll out a policy; coverage is averaged over every step of every episode."""
    cells = (np.arange(grid) + 0.5) * env.cfg.area_size / grid
    gx, gy = np.meshgrid(cells, cells)
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])
    heat = np.zeros(grid * grid, dtype=np.float64)
    step_coverages: list[float] = []
    final_coverages: list[float] = []
    errors: list[float] = []
    for _ in range(episodes):
        obs = env.reset()
        info = {"coverage": env.prev_coverage}
        for _ in range(env.cfg.drl_episode_len):
            obs, _, done, info = env.step(policy(env, obs))
            step_coverages.append(info["coverage"])
            errors.append(info["prediction_error_m"])
            heat += env.coverage_mask(grid_points)
            if done:
                break
        final_coverages.append(info["coverage"])
    return PolicyResult(
        mean_coverage=float(np.mean(step_coverages)),
        final_coverage=float(np.mean(final_coverages)),
        mean_prediction_error_m=float(np.mean(errors)),
        heatmap=(heat / max(1, len(step_coverages))).reshape(grid, grid),
    )


def run_baseline_policy(env: UAVCoverageEnv, policy: str, episodes: int = 10) -> float:
    if policy not in BASELINES:
        raise ValueError(f"Unknown baseline policy: {policy}")
    return evaluate_policy(env, BASELINES[policy], episodes=episodes).mean_coverage


def plot_training_curves(rewards: np.ndarray, coverages: np.ndarray, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(rewards, alpha=0.35, label="raw")
    axes[0].plot(moving_average(rewards), label="moving avg")
    axes[0].set_title("Episode Reward")
    axes[0].set_xlabel("Episode")
    axes[0].legend()
    axes[1].plot(coverages, alpha=0.35, label="raw")
    axes[1].plot(moving_average(coverages), label="moving avg")
    axes[1].set_title("User Coverage")
    axes[1].set_xlabel("Episode")
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
    ax.set_ylim(0.0, max(100.0, max(vals, default=0.0) + 5.0))
    ax.set_title("Policy Coverage Comparison")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_coverage_heatmap(heatmap: np.ndarray, area_size: float, title: str, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(heatmap, origin="lower", extent=(0, area_size, 0, area_size), vmin=0.0, vmax=1.0, cmap="viridis")
    fig.colorbar(image, ax=ax, label="Fraction of time covered")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_summary(
    path: str | Path,
    predictor_rmse: float,
    maddpg_coverage: float,
    greedy_coverage: float,
    static_coverage: float,
    random_coverage: float | None = None,
    center_rmse: float | None = None,
    position_source: str = "oracle",
    prediction_error_m: float | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["=== RESULTS SUMMARY ===", f"Predictor RMSE : {predictor_rmse:.2f} m"]
    if center_rmse is not None:
        lines.append(f"Centre-guess RMSE (floor to beat): {center_rmse:.2f} m")
    lines.append(f"UAV observations use: {position_source} user positions")
    if prediction_error_m is not None:
        lines.append(f"Mean in-env position error: {prediction_error_m:.2f} m")
    lines += [
        f"MADDPG coverage: {maddpg_coverage:.1%}",
        f"vs Greedy      : {(maddpg_coverage - greedy_coverage) * 100:+.1f} pts",
        f"vs Static hover: {(maddpg_coverage - static_coverage) * 100:+.1f} pts",
    ]
    if random_coverage is not None:
        lines.append(f"vs Random walk : {(maddpg_coverage - random_coverage) * 100:+.1f} pts")
    lines.append("(coverage = mean over every step of the evaluation episodes, no exploration noise)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
