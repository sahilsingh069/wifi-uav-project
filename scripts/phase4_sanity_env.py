from __future__ import annotations

import argparse

import numpy as np

from wifi_uav.config import get_config
from wifi_uav.uav_env import UAVCoverageEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="smoke", choices=["smoke", "medium", "full"])
    args = parser.parse_args()

    cfg = get_config(args.preset)
    env = UAVCoverageEnv(cfg, seed=cfg.seed)
    obs = env.reset()
    print(f"Observation dim: {len(obs[0])}")
    print(f"Number of agents: {cfg.n_uavs}")
    for step in range(3):
        actions = [np.random.uniform(-1.0, 1.0, env.act_dim).astype(np.float32) for _ in range(cfg.n_uavs)]
        _, reward, done, info = env.step(actions)
        print(f"Step {step}: coverage={info['coverage']:.1%}, reward={reward:.3f}, done={done}")


if __name__ == "__main__":
    main()
