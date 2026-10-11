from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import re

import numpy as np
import torch

from wifi_uav.device import best_device
from wifi_uav.config import apply_overrides, get_config
from wifi_uav.evaluate import (
    BASELINES,
    evaluate_policy,
    load_maddpg,
    maddpg_policy,
    make_env,
    plot_baselines,
    plot_coverage_heatmap,
    plot_training_curves,
    write_summary,
)


def _read_metric(text: str, key: str) -> float | None:
    match = re.search(rf"^{key}=([0-9.]+)", text, flags=re.MULTILINE)
    return float(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="smoke", choices=["smoke", "medium", "full"])
    parser.add_argument("--positions", choices=["predicted", "oracle"], default=None)
    parser.add_argument("--episodes", type=int, default=None, help="evaluation episodes per policy")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any config field")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="demo mode: print the comparison but keep existing result files untouched",
    )
    args = parser.parse_args()

    cfg = apply_overrides(get_config(args.preset), args.set)
    positions = args.positions or cfg.position_source
    episodes = args.episodes or (2 if args.preset == "smoke" else 50)
    device = best_device(allow_mps=False)
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(tempfile.mkdtemp()) if args.no_save else cfg.result_dir

    rewards_path = cfg.result_dir / "maddpg_rewards.npy"
    coverages_path = cfg.result_dir / "maddpg_coverages.npy"
    if rewards_path.exists() and coverages_path.exists():
        plot_training_curves(np.load(rewards_path), np.load(coverages_path), out_dir / "training_curves.png")

    # Every policy sees the same held-out user trajectories and UAV start positions.
    eval_seed = cfg.seed + 2000

    def fresh_env():
        return make_env(cfg, seed=eval_seed, use_predictor=positions == "predicted", device=device)

    env = fresh_env()
    model = load_maddpg(cfg, env, device)

    print(f"Evaluating every policy on the same {episodes} held-out episodes ({positions} positions)...", flush=True)
    results = {"MADDPG": evaluate_policy(env, maddpg_policy(model), episodes=episodes)}
    print(f"  {'MADDPG':<12} coverage {results['MADDPG'].mean_coverage:.1%}", flush=True)
    for name, policy in BASELINES.items():
        results[name] = evaluate_policy(fresh_env(), policy, episodes=episodes)
        print(f"  {name:<12} coverage {results[name].mean_coverage:.1%}", flush=True)
    print(flush=True)

    scores = {name: result.mean_coverage for name, result in results.items()}
    plot_baselines(scores, out_dir / "baseline_comparison.png")
    plot_coverage_heatmap(
        results["MADDPG"].heatmap,
        cfg.area_size,
        f"MADDPG coverage heatmap ({positions} positions)",
        out_dir / "coverage_heatmap.png",
    )

    predictor_summary = cfg.result_dir / "predictor_summary.txt"
    text = predictor_summary.read_text(encoding="utf-8") if predictor_summary.exists() else ""
    write_summary(
        out_dir / "final_summary.txt",
        predictor_rmse=_read_metric(text, "RMSE_meters") or 0.0,
        maddpg_coverage=scores["MADDPG"],
        greedy_coverage=scores["Greedy"],
        static_coverage=scores["Static"],
        random_coverage=scores["Random"],
        local_greedy_coverage=scores["Local greedy"],
        center_rmse=_read_metric(text, "Center_guess_RMSE_meters"),
        position_source=positions,
        prediction_error_m=results["MADDPG"].mean_prediction_error_m if positions == "predicted" else None,
        distances_m={name: result.mean_distance_m for name, result in results.items()},
    )
    print((out_dir / "final_summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
