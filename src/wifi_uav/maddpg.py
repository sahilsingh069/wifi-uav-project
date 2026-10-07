from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn


class ReplayBuffer:
    def __init__(self, capacity: int, n_agents: int, obs_dim: int, act_dim: int, seed: int = 42):
        self.items: deque[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]] = deque(maxlen=capacity)
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
        reward: float | list[float] | np.ndarray,
        next_obs: list[np.ndarray],
        done: bool,
    ) -> None:
        """``reward`` is either one team reward or one reward per agent."""
        self.items.append(
            (
                np.asarray(obs, dtype=np.float32),
                np.asarray(actions, dtype=np.float32),
                np.broadcast_to(np.asarray(reward, dtype=np.float32), (self.n_agents,)).copy(),
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
            "rewards": torch.tensor(np.stack(rewards), dtype=torch.float32),
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
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))

    def forward_with_preactivation(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre = self.net(obs)
        return torch.tanh(pre), pre


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


class SharedMADDPG:
    """MADDPG with parameter sharing for homogeneous agents.

    One actor acts for every UAV from that UAV's own (egocentric) observation. One centralised
    critic scores agent i by seeing everyone's observations and actions, reordered so agent i's
    come first; that lets a single network serve as every agent's critic. Each environment step
    therefore yields ``n_agents`` training samples for both networks.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_agents: int,
        lr: float = 1e-3,
        actor_lr: float = 3e-4,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.device = torch.device(device)
        self.actor = ActorNet(obs_dim, act_dim).to(self.device)
        self.actor_target = ActorNet(obs_dim, act_dim).to(self.device)
        self.critic = CriticNet(obs_dim * n_agents, act_dim * n_agents).to(self.device)
        self.critic_target = CriticNet(obs_dim * n_agents, act_dim * n_agents).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=lr)
        # perm[i] = [i, then every other agent in index order]
        self.perm = torch.tensor(
            [[i] + [j for j in range(n_agents) if j != i] for i in range(n_agents)], device=self.device
        )

    def select_actions(self, obs: list[np.ndarray], noise: float = 0.0) -> list[np.ndarray]:
        with torch.no_grad():
            obs_t = torch.tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
            actions = self.actor(obs_t).cpu().numpy()
        if noise > 0.0:
            actions = actions + np.random.normal(0.0, noise, size=actions.shape)
        return [a.astype(np.float32) for a in np.clip(actions, -1.0, 1.0)]

    def _agent_centric(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n, d) -> (n * B, n * d), block i holding every sample reordered for agent i."""
        batch = x.shape[0]
        return x[:, self.perm].permute(1, 0, 2, 3).reshape(self.n_agents * batch, -1)

    def update(
        self,
        batch: dict[str, torch.Tensor],
        gamma: float = 0.95,
        tau: float = 0.01,
        preact_reg: float = 1e-3,
    ) -> dict[str, float]:
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        # (B, n) -> (n * B, 1) in the same agent-major order as _agent_centric.
        rewards = batch["rewards"].to(self.device).T.reshape(-1, 1)
        dones = batch["dones"].to(self.device).repeat(self.n_agents, 1)

        critic_obs = self._agent_centric(obs)
        critic_next_obs = self._agent_centric(next_obs)
        with torch.no_grad():
            next_actions = self._agent_centric(self.actor_target(next_obs))
            target_q = rewards + gamma * (1.0 - dones) * self.critic_target(critic_next_obs, next_actions)

        q_value = self.critic(critic_obs, self._agent_centric(actions))
        critic_loss = nn.functional.mse_loss(q_value, target_q)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # Each agent re-plans only its own action; teammates keep their replayed actions.
        own_actions, preact = self.actor.forward_with_preactivation(obs)
        joint = actions[:, self.perm].permute(1, 0, 2, 3).clone()  # (n, B, n, act)
        joint[:, :, 0, :] = own_actions.permute(1, 0, 2)
        joint = joint.reshape(self.n_agents * obs.shape[0], -1)
        # The pre-activation penalty keeps the actor out of tanh saturation; otherwise it learns
        # bang-bang actions that pin UAVs to the area edge and the altitude ceiling.
        actor_loss = -self.critic(critic_obs, joint).mean() + preact_reg * (preact**2).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        soft_update(self.actor_target, self.actor, tau)
        soft_update(self.critic_target, self.critic, tau)
        return {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item())}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"actor": self.actor.state_dict(), "obs_dim": self.obs_dim, "act_dim": self.act_dim, "n_agents": self.n_agents},
            path,
        )

    @classmethod
    def load_actor(cls, path: str | Path, device: str = "cpu") -> "SharedMADDPG":
        state = torch.load(path, map_location=device)
        model = cls(state["obs_dim"], state["act_dim"], state["n_agents"], device=device)
        model.actor.load_state_dict(state["actor"])
        model.actor.eval()
        return model
