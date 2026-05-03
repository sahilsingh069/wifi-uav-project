from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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


def run_baseline_policy(env: UAVCoverageEnv, policy: str, episodes: int = 10) -> float:
    coverages: list[float] = []
    for _ in range(episodes):
        env.reset()
        last_coverage = 0.0
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

            _, _, done, info = env.step(actions)
            last_coverage = info["coverage"]
            if done:
                break
        coverages.append(last_coverage)
    return float(np.mean(coverages))


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


def write_summary(
    path: str | Path,
    predictor_rmse: float,
    maddpg_coverage: float,
    greedy_coverage: float,
    static_coverage: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "=== RESULTS SUMMARY ===\n"
        f"Predictor RMSE : {predictor_rmse:.2f} m\n"
        f"MADDPG coverage: {maddpg_coverage:.1%}\n"
        f"vs Greedy      : {(maddpg_coverage - greedy_coverage) * 100:.1f}%\n"
        f"vs Static hover: {(maddpg_coverage - static_coverage) * 100:.1f}%\n"
    )
    path.write_text(text, encoding="utf-8")
