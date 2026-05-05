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
            progress = ep / max(1, cfg.drl_episodes - 1)
            noise = max(cfg.maddpg_noise_end, cfg.maddpg_noise_start * (1.0 - progress))
            actions = [agent.select_action(obs[idx], noise=noise) for idx, agent in enumerate(agents)]
            next_obs, reward, done, info = env.step(actions)
            buffer.push(obs, actions, reward, next_obs, done)
            batch_size = min(cfg.batch_size, 256)
            if len(buffer) >= max(cfg.maddpg_warmup_steps, batch_size) and len(buffer) % cfg.maddpg_update_every == 0:
                batch = buffer.sample(batch_size)
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
    (cfg.result_dir / "maddpg_training_summary.txt").write_text(
        "\n".join(
            [
                f"episodes={cfg.drl_episodes}",
                f"episode_len={cfg.drl_episode_len}",
                f"warmup_steps={cfg.maddpg_warmup_steps}",
                f"update_every={cfg.maddpg_update_every}",
                f"final_reward={rewards[-1]:.4f}",
                f"final_coverage={coverages[-1]:.4f}",
                f"avg_last_10_reward={np.mean(rewards[-10:]):.4f}",
                f"avg_last_10_coverage={np.mean(coverages[-10:]):.4f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for idx, agent in enumerate(agents):
        torch.save(agent.actor.state_dict(), cfg.checkpoint_dir / f"maddpg_actor_{idx}.pt")


if __name__ == "__main__":
    main()
