from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from wifi_uav.config import apply_overrides, get_config
from wifi_uav.evaluate import BASELINES, load_maddpg, maddpg_policy, make_env
from wifi_uav.replay import record_episode, scene_metadata, write_replays


def main() -> None:
    parser = argparse.ArgumentParser(description="Record evaluation episodes as JSON for the web viewer.")
    parser.add_argument("--preset", default="medium", choices=["smoke", "medium", "full"])
    parser.add_argument("--episodes", type=int, default=3, help="episodes recorded per policy and position mode")
    parser.add_argument("--steps", type=int, default=None, help="steps per episode (defaults to the preset)")
    parser.add_argument("--output", default="web/data/replays.json")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any config field")
    args = parser.parse_args()

    cfg = apply_overrides(get_config(args.preset), args.set)
    if args.steps is not None:
        cfg = replace(cfg, drl_episode_len=args.steps)
    steps = cfg.drl_episode_len

    policies = dict(BASELINES)
    try:
        model = load_maddpg(cfg, make_env(cfg, seed=0, use_predictor=False))
        policies = {"MADDPG": maddpg_policy(model), **policies}
    except (FileNotFoundError, RuntimeError) as exc:
        # RuntimeError: checkpoints were trained for a different preset (state_dict shape mismatch).
        print(f"Skipping MADDPG: {str(exc).splitlines()[0]}")

    modes = ["oracle"]
    try:
        make_env(cfg, seed=0, use_predictor=True)
        modes = ["predicted", "oracle"]
    except FileNotFoundError as exc:
        print(f"Skipping predicted positions: {exc}")

    runs = []
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for mode in modes:
        for name, policy in policies.items():
            coverages, distances = [], []
            for episode in range(args.episodes):
                # Same seed per episode index: every policy faces identical users and UAV start points.
                env = make_env(cfg, seed=cfg.seed + 3000 + episode, use_predictor=mode == "predicted")
                frames = record_episode(env, policy, steps)
                runs.append({"policy": name, "positions": mode, "episode": episode, "frames": frames})
                coverages.append(float(np.mean(frames["coverage"][1:])))
                distances.append(float(np.mean(frames["distance_m"][1:])))
            summary.setdefault(mode, {})[name] = {
                "coverage": round(float(np.mean(coverages)), 4),
                "distance_m": round(float(np.mean(distances)), 1),
            }
            print(f"{mode:9s} {name:7s} coverage {np.mean(coverages):.1%}  flown/step {np.mean(distances):.1f} m")

    meta = scene_metadata(cfg, make_env(cfg, seed=0, use_predictor=False), steps)
    meta.update({"preset": args.preset, "episodes": args.episodes, "policies": list(policies), "modes": modes})
    predictor_summary = cfg.result_dir / "predictor_summary.txt"
    if predictor_summary.exists():
        for line in predictor_summary.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key in {"RMSE_meters", "Center_guess_RMSE_meters"}:
                meta[key.lower()] = round(float(value), 1)

    write_replays(args.output, meta, runs, summary)
    size_kb = Path(args.output).stat().st_size / 1024
    print(f"Wrote {args.output} ({len(runs)} episodes, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
