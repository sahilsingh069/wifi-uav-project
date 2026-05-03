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
    rewards_path = cfg.result_dir / "maddpg_rewards.npy"
    coverages_path = cfg.result_dir / "maddpg_coverages.npy"
    rewards = np.load(rewards_path) if rewards_path.exists() else np.zeros(1, dtype=np.float32)
    coverages = np.load(coverages_path) if coverages_path.exists() else np.zeros(1, dtype=np.float32)
    plot_training_curves(rewards, coverages, cfg.result_dir / "training_curves.png")

    env = UAVCoverageEnv(cfg, seed=cfg.seed)
    baseline_episodes = 2 if args.preset == "smoke" else 50
    scores = {
        "MADDPG": float(coverages[-1]) if len(coverages) else 0.0,
        "Greedy": run_baseline_policy(env, "greedy", episodes=baseline_episodes),
        "Random": run_baseline_policy(env, "random", episodes=baseline_episodes),
        "Static": run_baseline_policy(env, "static", episodes=baseline_episodes),
    }
    plot_baselines(scores, cfg.result_dir / "baseline_comparison.png")

    predictor_summary = cfg.result_dir / "predictor_summary.txt"
    rmse = 0.0
    if predictor_summary.exists():
        match = re.search(r"RMSE_meters=([0-9.]+)", predictor_summary.read_text(encoding="utf-8"))
        rmse = float(match.group(1)) if match else 0.0

    write_summary(cfg.result_dir / "final_summary.txt", rmse, scores["MADDPG"], scores["Greedy"], scores["Static"])
    print((cfg.result_dir / "final_summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
