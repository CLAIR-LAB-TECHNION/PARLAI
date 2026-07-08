"""DQN trainer carried over from Tutorial 07 for the sample-efficiency comparison.

This is the replay-buffer + target-network DQN built step by step in Tut07
(``train_dqn_full``), with one instrumentation change: the returned history
records the cumulative number of *training* environment steps consumed at the
end of each episode, so the notebook can plot learning curves against
environment steps rather than episodes. Evaluation rollouts are greedy and are
not counted, following the usual convention that evaluation interaction is free.
"""

import random
from collections import deque

import numpy as np
import torch
from torch import nn

from .common import set_seed


def linear_schedule(t, eps_start=1.0, eps_end=0.05, decay_episodes=100):
    """Linear epsilon decay from Tut07: eps falls from start to end over
    ``decay_episodes`` episodes and stays flat afterwards."""
    return max(eps_end, eps_start - (eps_start - eps_end) * t / decay_episodes)


class QNet(nn.Module):
    """Two-hidden-layer MLP mapping a state vector to one Q-value per action
    (identical to the Q-network of Tut07)."""

    def __init__(self, in_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """FIFO transition buffer with uniform minibatch sampling (from Tut07)."""

    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_next, d = zip(*batch)
        return (
            torch.from_numpy(np.asarray(s, dtype=np.float32)),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.from_numpy(np.asarray(s_next, dtype=np.float32)),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


def _greedy_return(env, qnet, seed):
    """Play one greedy episode and return its (undiscounted) return."""
    obs, _ = env.reset(seed=seed)
    total = 0.0
    done = False
    while not done:
        with torch.no_grad():
            a = int(qnet(torch.from_numpy(np.asarray(obs, dtype=np.float32))).argmax().item())
        obs, r, term, trun, _ = env.step(a)
        total += r
        done = term or trun
    return total


def train_dqn(env, n_episodes=300, gamma=0.99, lr=1e-3, hidden_dim=64,
              eps_start=0.1, eps_end=0.1, eps_decay_episodes=800,
              buffer_size=1000, warmup=200, train_every=1, batch_size=64,
              target_sync_freq=100, eval_every=1, seed=0, progress=None):
    """Replay-buffer + target-network DQN (Tut07's ``train_dqn_full``).

    Args:
        env: gymnasium env with a Box observation space and Discrete actions.
        n_episodes: number of training episodes.
        gamma, lr, hidden_dim: discount, Adam learning rate, MLP hidden width.
        eps_start, eps_end, eps_decay_episodes: linear exploration schedule.
        buffer_size, warmup, train_every, batch_size: replay-buffer settings;
            training starts once the buffer holds ``max(warmup, batch_size)``
            transitions.
        target_sync_freq: gradient steps between target-network syncs.
        eval_every: run one greedy evaluation episode every this many episodes.
        seed: seed for the networks, exploration, and environment resets.
        progress: optional tqdm-like iterator wrapper for the episode loop.

    Returns:
        (history, qnet) where history is a list of
        ``(cumulative_env_steps, eval_return)`` tuples, one per evaluation.
    """
    set_seed(seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    qnet = QNet(obs_dim, n_actions, hidden=hidden_dim)
    qnet_target = QNet(obs_dim, n_actions, hidden=hidden_dim)
    qnet_target.load_state_dict(qnet.state_dict())
    optimizer = torch.optim.Adam(qnet.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size)

    history = []
    env_steps = 0  # training interaction only; greedy evals are not counted
    grad_steps = 0
    episodes = range(n_episodes) if progress is None else progress(range(n_episodes))
    for episode in episodes:
        obs, _ = env.reset(seed=seed + episode)
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        epsilon = linear_schedule(episode, eps_start, eps_end, eps_decay_episodes)
        term = trun = False
        while not (term or trun):
            # epsilon-greedy action from the online network
            if np.random.rand() < epsilon:
                a = np.random.randint(n_actions)
            else:
                with torch.no_grad():
                    a = int(qnet(x).argmax().item())
            next_obs, r, term, trun, _ = env.step(a)
            env_steps += 1
            next_x = torch.from_numpy(np.asarray(next_obs, dtype=np.float32))
            buffer.push(x, a, r, next_x, term)
            x = next_x
            if len(buffer) >= max(warmup, batch_size) and len(buffer) % train_every == 0:
                s, ac, rw, s_next, t = buffer.sample(batch_size)
                # bootstrap target from the frozen target network
                with torch.no_grad():
                    target = rw + gamma * qnet_target(s_next).max(dim=1).values * (1 - t)
                pred = qnet(s).gather(1, ac.unsqueeze(1)).squeeze(1)
                loss = (pred - target).pow(2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                grad_steps += 1
                if grad_steps % target_sync_freq == 0:
                    qnet_target.load_state_dict(qnet.state_dict())
        if episode % eval_every == 0:
            ret = _greedy_return(env, qnet, seed=10_000 + seed + episode)
            history.append((env_steps, ret))
    return history, qnet
