import numpy as np
import torch

from wifi_uav.maddpg import ActorNet, CriticNet, ReplayBuffer


def test_replay_buffer_samples_expected_shapes():
    buffer = ReplayBuffer(capacity=10, n_agents=3, obs_dim=17, act_dim=3, seed=1)
    obs = [np.zeros(17, dtype=np.float32) for _ in range(3)]
    acts = [np.zeros(3, dtype=np.float32) for _ in range(3)]
    for _ in range(5):
        buffer.push(obs, acts, 1.0, obs, False)

    batch = buffer.sample(4)

    assert batch["obs"].shape == (4, 3, 17)
    assert batch["actions"].shape == (4, 3, 3)
    assert batch["rewards"].shape == (4, 1)


def test_actor_and_critic_forward_shapes():
    actor = ActorNet(obs_dim=17, act_dim=3)
    critic = CriticNet(total_obs_dim=51, total_act_dim=9)

    action = actor(torch.randn(2, 17))
    q = critic(torch.randn(2, 51), torch.randn(2, 9))

    assert action.shape == (2, 3)
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)
    assert q.shape == (2, 1)
