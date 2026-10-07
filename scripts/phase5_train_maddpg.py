from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import torch

from wifi_uav.config import apply_overrides, get_config
from wifi_uav.evaluate import make_env
from wifi_uav.maddpg import ReplayBuffer, SharedMADDPG


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="smoke", choices=["smoke", "medium", "full"])
    parser.add_argument("--positions", choices=["predicted", "oracle"], default=None)
    parser.add_argument("--episodes", type=int, default=None, help="override the preset's drl_episodes")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any config field")
    args = parser.parse_args()

    cfg = get_config(args.preset)
    if args.episodes is not None:
        cfg = replace(cfg, drl_episodes=args.episodes)
    cfg = apply_overrides(cfg, args.set)
    positions = args.positions or cfg.position_source
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Training trajectories use a different seed from the RF dataset and from evaluation.
    env = make_env(cfg, seed=cfg.seed + 1000, use_predictor=positions == "predicted", device=device)
    obs = env.reset()
    obs_dim = len(obs[0])
    print(f"Training with {positions} user positions on {device}")
    model = SharedMADDPG(
        obs_dim,
        env.act_dim,
        cfg.n_uavs,
        lr=cfg.learning_rate,
        actor_lr=cfg.maddpg_actor_lr,
        device=device,
    )
    buffer = ReplayBuffer(100_000, cfg.n_uavs, obs_dim, env.act_dim, seed=cfg.seed)
    rewards: list[float] = []
    coverages: list[float] = []

    for ep in range(cfg.drl_episodes):
        obs = env.reset()
        ep_reward = 0.0
        step_covs: list[float] = []
        for _ in range(cfg.drl_episode_len):
            progress = ep / max(1, cfg.drl_episodes - 1)
            noise = max(cfg.maddpg_noise_end, cfg.maddpg_noise_start * (1.0 - progress))
            actions = model.select_actions(obs, noise=noise)
            next_obs, reward, done, info = env.step(actions)
            buffer.push(obs, actions, info["agent_rewards"], next_obs, done)
            batch_size = min(cfg.batch_size, 256)
            if len(buffer) >= max(cfg.maddpg_warmup_steps, batch_size) and len(buffer) % cfg.maddpg_update_every == 0:
                for _ in range(cfg.maddpg_updates_per_step):
                    model.update(buffer.sample(batch_size), gamma=cfg.maddpg_gamma)
            obs = next_obs
            ep_reward += reward
            step_covs.append(info["coverage"])
            if done:
                break
        rewards.append(ep_reward)
        coverages.append(float(np.mean(step_covs)))
        print(f"Episode {ep + 1}: reward={ep_reward:.3f}, mean coverage={coverages[-1]:.1%}, replay={len(buffer)}")

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.result_dir / "maddpg_rewards.npy", np.asarray(rewards, dtype=np.float32))
    np.save(cfg.result_dir / "maddpg_coverages.npy", np.asarray(coverages, dtype=np.float32))
    (cfg.result_dir / "maddpg_training_summary.txt").write_text(
        "\n".join(
            [
                f"positions={positions}",
                f"episodes={cfg.drl_episodes}",
                f"episode_len={cfg.drl_episode_len}",
                f"warmup_steps={cfg.maddpg_warmup_steps}",
                f"update_every={cfg.maddpg_update_every}",
                f"updates_per_step={cfg.maddpg_updates_per_step}",
                f"gamma={cfg.maddpg_gamma}",
                f"actor_lr={cfg.maddpg_actor_lr}",
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
    model.save(cfg.checkpoint_dir / "maddpg_actor.pt")


if __name__ == "__main__":
    main()
