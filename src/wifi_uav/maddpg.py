from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class ReplayBuffer:
    def __init__(self, capacity: int, n_agents: int, obs_dim: int, act_dim: int, seed: int = 42):
        self.items: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque(maxlen=capacity)
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def push(
        self,
        obs: list[np.ndarray],
        actions: list[np.ndarray],
        reward: float,
        next_obs: list[np.ndarray],
        done: bool,
    ) -> None:
        self.items.append(
            (
                np.asarray(obs, dtype=np.float32),
                np.asarray(actions, dtype=np.float32),
                float(reward),
                np.asarray(next_obs, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if batch_size > len(self.items):
            raise ValueError(f"batch_size {batch_size} exceeds replay size {len(self.items)}")
        indices = self.rng.choice(len(self.items), size=batch_size, replace=False)
        obs, actions, rewards, next_obs, dones = zip(*(self.items[idx] for idx in indices))
        return {
            "obs": torch.tensor(np.stack(obs), dtype=torch.float32),
            "actions": torch.tensor(np.stack(actions), dtype=torch.float32),
            "rewards": torch.tensor(np.asarray(rewards, dtype=np.float32)[:, None], dtype=torch.float32),
            "next_obs": torch.tensor(np.stack(next_obs), dtype=torch.float32),
            "dones": torch.tensor(np.asarray(dones, dtype=np.float32)[:, None], dtype=torch.float32),
        }


class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CriticNet(nn.Module):
    def __init__(self, total_obs_dim: int, total_act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_act_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, all_obs: torch.Tensor, all_actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([all_obs, all_actions], dim=1))


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


@dataclass
class TrainingStats:
    rewards: list[float]
    coverages: list[float]


class MADDPGAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        total_obs_dim: int,
        total_act_dim: int,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.actor = ActorNet(obs_dim, act_dim).to(self.device)
        self.actor_target = ActorNet(obs_dim, act_dim).to(self.device)
        self.critic = CriticNet(total_obs_dim, total_act_dim).to(self.device)
        self.critic_target = CriticNet(total_obs_dim, total_act_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=lr)

    def select_action(self, obs: np.ndarray, noise: float = 0.0) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.actor(obs_t).squeeze(0).cpu().numpy()
        if noise > 0.0:
            action = action + np.random.normal(0.0, noise, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def update_agents(
    agents: list[MADDPGAgent],
    batch: dict[str, torch.Tensor],
    gamma: float = 0.95,
    tau: float = 0.01,
) -> None:
    device = agents[0].device
    obs = batch["obs"].to(device)
    actions = batch["actions"].to(device)
    rewards = batch["rewards"].to(device)
    next_obs = batch["next_obs"].to(device)
    dones = batch["dones"].to(device)

    batch_size, n_agents, obs_dim = obs.shape
    act_dim = actions.shape[-1]
    flat_obs = obs.reshape(batch_size, n_agents * obs_dim)
    flat_actions = actions.reshape(batch_size, n_agents * act_dim)

    with torch.no_grad():
        next_actions = torch.stack([agent.actor_target(next_obs[:, idx, :]) for idx, agent in enumerate(agents)], dim=1)
        flat_next_obs = next_obs.reshape(batch_size, n_agents * obs_dim)
        flat_next_actions = next_actions.reshape(batch_size, n_agents * act_dim)

    for idx, agent in enumerate(agents):
        with torch.no_grad():
            target_q = rewards + gamma * (1.0 - dones) * agent.critic_target(flat_next_obs, flat_next_actions)

        q_value = agent.critic(flat_obs, flat_actions)
        critic_loss = nn.functional.mse_loss(q_value, target_q)
        agent.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
        agent.critic_opt.step()

        current_actions = actions.clone()
        current_actions[:, idx, :] = agent.actor(obs[:, idx, :])
        actor_loss = -agent.critic(flat_obs, current_actions.reshape(batch_size, n_agents * act_dim)).mean()
        agent.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
        agent.actor_opt.step()

        soft_update(agent.actor_target, agent.actor, tau)
        soft_update(agent.critic_target, agent.critic, tau)
