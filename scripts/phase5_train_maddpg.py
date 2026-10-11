from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace

import numpy as np
import torch

from wifi_uav.device import best_device
from wifi_uav.config import apply_overrides, get_config
from wifi_uav.evaluate import make_env
from wifi_uav.maddpg import ReplayBuffer, SharedMADDPG


def log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="smoke", choices=["smoke", "medium", "full"])
    parser.add_argument("--positions", choices=["predicted", "oracle"], default=None)
    parser.add_argument("--episodes", type=int, default=None, help="override the preset's drl_episodes")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any config field")
    parser.add_argument("--log-every", type=int, default=1, help="print one line every N episodes")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="demo mode: train and print live, but keep existing checkpoints and results untouched",
    )
    args = parser.parse_args()

    cfg = get_config(args.preset)
    if args.episodes is not None:
        cfg = replace(cfg, drl_episodes=args.episodes)
    cfg = apply_overrides(cfg, args.set)
    positions = args.positions or cfg.position_source
    device = best_device(allow_mps=False)
    # Training trajectories use a different seed from the RF dataset and from evaluation.
    env = make_env(cfg, seed=cfg.seed + 1000, use_predictor=positions == "predicted", device=device)
    obs = env.reset()
    obs_dim = len(obs[0])
    log(f"Training {cfg.n_uavs} UAVs for {cfg.drl_episodes} episodes x {cfg.drl_episode_len} steps, "
        f"{positions} user positions, device {device}")
    log(f"No learning for the first {cfg.maddpg_warmup_steps} steps (replay warm-up); exploration noise "
        f"{cfg.maddpg_noise_start} -> {cfg.maddpg_noise_end}")
    if args.no_save:
        log("Demo mode (--no-save): existing checkpoints and results will not be changed.")
    log(f"{'episode':>7}  {'noise':>5}  {'reward':>7}  {'coverage':>8}  {'last-10 avg':>11}  "
        f"{'critic_loss':>11}  {'actor_loss':>10}  {'replay':>6}  {'time':>5}")
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
    history_rows: list[list[float]] = []

    for ep in range(cfg.drl_episodes):
        obs = env.reset()
        ep_reward = 0.0
        step_covs: list[float] = []
        losses: list[dict[str, float]] = []
        started = time.perf_counter()
        for _ in range(cfg.drl_episode_len):
            progress = ep / max(1, cfg.drl_episodes - 1)
            noise = max(cfg.maddpg_noise_end, cfg.maddpg_noise_start * (1.0 - progress))
            actions = model.select_actions(obs, noise=noise)
            next_obs, reward, done, info = env.step(actions)
            buffer.push(obs, actions, info["agent_rewards"], next_obs, done)
            batch_size = min(cfg.batch_size, 256)
            if len(buffer) >= max(cfg.maddpg_warmup_steps, batch_size) and len(buffer) % cfg.maddpg_update_every == 0:
                for _ in range(cfg.maddpg_updates_per_step):
                    losses.append(model.update(buffer.sample(batch_size), gamma=cfg.maddpg_gamma))
            obs = next_obs
            ep_reward += reward
            step_covs.append(info["coverage"])
            if done:
                break
        rewards.append(ep_reward)
        coverages.append(float(np.mean(step_covs)))
        critic_loss = float(np.mean([item["critic_loss"] for item in losses])) if losses else float("nan")
        actor_loss = float(np.mean([item["actor_loss"] for item in losses])) if losses else float("nan")
        seconds = time.perf_counter() - started
        history_rows.append([ep + 1, noise, ep_reward, coverages[-1], critic_loss, actor_loss, len(buffer), seconds])
        if (ep + 1) % args.log_every == 0 or ep == 0 or ep + 1 == cfg.drl_episodes:
            log(
                f"{ep + 1:>7}  {noise:>5.2f}  {ep_reward:>7.2f}  {coverages[-1]:>8.1%}  "
                f"{np.mean(coverages[-10:]):>11.1%}  {critic_loss:>11.4f}  {actor_loss:>10.4f}  "
                f"{len(buffer):>6}  {seconds:>4.1f}s"
            )

    if args.no_save:
        log(f"Done. Last-10-episode mean coverage (with exploration noise): {np.mean(coverages[-10:]):.1%}")
        return

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.result_dir / "maddpg_rewards.npy", np.asarray(rewards, dtype=np.float32))
    np.save(cfg.result_dir / "maddpg_coverages.npy", np.asarray(coverages, dtype=np.float32))
    with (cfg.result_dir / "maddpg_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "noise", "reward", "coverage", "critic_loss", "actor_loss", "replay", "seconds"])
        writer.writerows([[f"{value:.6g}" for value in row] for row in history_rows])
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
    log(f"Saved {cfg.checkpoint_dir / 'maddpg_actor.pt'} and {cfg.result_dir / 'maddpg_history.csv'}")


if __name__ == "__main__":
    main()
