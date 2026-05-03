from __future__ import annotations

import argparse

import numpy as np
import torch

from wifi_uav.config import get_config
from wifi_uav.maddpg import MADDPGAgent, ReplayBuffer, update_agents
from wifi_uav.uav_env import UAVCoverageEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="smoke", choices=["smoke", "medium", "full"])
    args = parser.parse_args()

    cfg = get_config(args.preset)
    env = UAVCoverageEnv(cfg, seed=cfg.seed)
    obs = env.reset()
    obs_dim = len(obs[0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agents = [
        MADDPGAgent(
            obs_dim,
            env.act_dim,
            obs_dim * cfg.n_uavs,
            env.act_dim * cfg.n_uavs,
            lr=cfg.learning_rate,
            device=device,
        )
        for _ in range(cfg.n_uavs)
    ]
    buffer = ReplayBuffer(100_000, cfg.n_uavs, obs_dim, env.act_dim, seed=cfg.seed)
    rewards: list[float] = []
    coverages: list[float] = []

    for ep in range(cfg.drl_episodes):
        obs = env.reset()
        ep_reward = 0.0
        last_cov = 0.0
        for _ in range(cfg.drl_episode_len):
            noise = max(0.05, 0.3 * (1.0 - ep / max(1, cfg.drl_episodes - 1)))
            actions = [agent.select_action(obs[idx], noise=noise) for idx, agent in enumerate(agents)]
            next_obs, reward, done, info = env.step(actions)
            buffer.push(obs, actions, reward, next_obs, done)
            if len(buffer) >= min(cfg.batch_size, 64):
                batch = buffer.sample(min(cfg.batch_size, 64))
                update_agents(agents, batch)
            obs = next_obs
            ep_reward += reward
            last_cov = info["coverage"]
            if done:
                break
        rewards.append(ep_reward)
        coverages.append(last_cov)
        print(f"Episode {ep + 1}: reward={ep_reward:.3f}, coverage={last_cov:.1%}, replay={len(buffer)}")

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.result_dir / "maddpg_rewards.npy", np.asarray(rewards, dtype=np.float32))
    np.save(cfg.result_dir / "maddpg_coverages.npy", np.asarray(coverages, dtype=np.float32))
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for idx, agent in enumerate(agents):
        torch.save(agent.actor.state_dict(), cfg.checkpoint_dir / f"maddpg_actor_{idx}.pt")


if __name__ == "__main__":
    main()
