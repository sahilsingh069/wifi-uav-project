from __future__ import annotations

import argparse

import numpy as np

from wifi_uav.config import get_config
from wifi_uav.evaluate import make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="smoke", choices=["smoke", "medium", "full"])
    parser.add_argument("--positions", choices=["predicted", "oracle"], default=None)
    args = parser.parse_args()

    cfg = get_config(args.preset)
    positions = args.positions or cfg.position_source
    env = make_env(cfg, seed=cfg.seed + 1000, use_predictor=positions == "predicted")
    obs = env.reset()
    print(f"Observation dim: {len(obs[0])}")
    print(f"Number of agents: {cfg.n_uavs}")
    print(f"User positions observed via: {positions}")
    print(f"Optimal altitude: {env.optimal_altitude:.1f} m, footprint there: {float(env.footprint_radius(env.optimal_altitude)):.1f} m")
    for step in range(3):
        actions = [env.rng.uniform(-1.0, 1.0, env.act_dim).astype(np.float32) for _ in range(cfg.n_uavs)]
        _, reward, done, info = env.step(actions)
        print(
            f"Step {step}: coverage={info['coverage']:.1%}, reward={reward:.3f}, "
            f"position_error={info['prediction_error_m']:.1f} m, done={done}"
        )


if __name__ == "__main__":
    main()
