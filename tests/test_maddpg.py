import numpy as np
import torch

from wifi_uav.maddpg import ActorNet, CriticNet, ReplayBuffer, SharedMADDPG


def test_replay_buffer_samples_expected_shapes():
    buffer = ReplayBuffer(capacity=10, n_agents=3, obs_dim=17, act_dim=3, seed=1)
    obs = [np.zeros(17, dtype=np.float32) for _ in range(3)]
    acts = [np.zeros(3, dtype=np.float32) for _ in range(3)]
    for _ in range(5):
        buffer.push(obs, acts, 1.0, obs, False)

    batch = buffer.sample(4)

    assert batch["obs"].shape == (4, 3, 17)
    assert batch["actions"].shape == (4, 3, 3)
    assert batch["rewards"].shape == (4, 3)


def test_actor_and_critic_forward_shapes():
    actor = ActorNet(obs_dim=17, act_dim=3)
    critic = CriticNet(total_obs_dim=51, total_act_dim=9)

    action = actor(torch.randn(2, 17))
    q = critic(torch.randn(2, 51), torch.randn(2, 9))

    assert action.shape == (2, 3)
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)
    assert q.shape == (2, 1)


def test_shared_maddpg_update_and_reload(tmp_path):
    model = SharedMADDPG(obs_dim=8, act_dim=3, n_agents=3)
    buffer = ReplayBuffer(capacity=50, n_agents=3, obs_dim=8, act_dim=3, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs = list(rng.normal(size=(3, 8)).astype(np.float32))
        acts = model.select_actions(obs, noise=0.3)
        buffer.push(obs, acts, rng.normal(size=3), obs, False)

    losses = model.update(buffer.sample(8))
    model.save(tmp_path / "actor.pt")
    reloaded = SharedMADDPG.load_actor(tmp_path / "actor.pt")

    assert np.isfinite(losses["critic_loss"]) and np.isfinite(losses["actor_loss"])
    obs = list(rng.normal(size=(3, 8)).astype(np.float32))
    assert np.allclose(model.select_actions(obs), reloaded.select_actions(obs), atol=1e-6)


def test_agent_centric_ordering_puts_own_agent_first():
    model = SharedMADDPG(obs_dim=1, act_dim=1, n_agents=3)
    x = torch.tensor([[[0.0], [1.0], [2.0]]])

    rows = model._agent_centric(x)

    assert rows.tolist() == [[0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [2.0, 0.0, 1.0]]
