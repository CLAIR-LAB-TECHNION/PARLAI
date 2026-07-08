"""Tabular Q-learning runner for the VaultEnv landscape demo.

The notebook uses this to contrast value-based and policy-based optimization
on the same tiny MDP: Q-learning (this module) reaches the optimal policy from
any initialization, while gradient ascent on the policy objective can be
caught by the environment's decoy attractor.
"""

import numpy as np


def tabular_q_learning(env, n_episodes=500, alpha=0.2, gamma=1.0, epsilon=0.2,
                       q_init_scale=1.0, seed=0):
    """Run tabular Q-learning (Tut06 style) on a small discrete-state env.

    Args:
        env: gymnasium env with ``Discrete`` observation and action spaces.
        n_episodes: number of training episodes.
        alpha: learning rate of the incremental Q update.
        gamma: discount factor (1.0 for the short episodic VaultEnv).
        epsilon: constant epsilon-greedy exploration rate.
        q_init_scale: the Q-table is initialized ~ N(0, q_init_scale^2); a
            nonzero scale lets the demo start from many *different* random
            initializations, mirroring the random policy-parameter starts.
        seed: seed for the table init, exploration, and environment resets.

    Returns:
        The learned Q-table, shape (n_states, n_actions).
    """
    rng = np.random.default_rng(seed)
    n_s = env.observation_space.n
    n_a = env.action_space.n
    # random init: the point of the demo is that the *starting point* does not
    # matter for Q-learning, unlike for policy-gradient ascent
    q = rng.normal(0.0, q_init_scale, size=(n_s, n_a))

    for ep in range(n_episodes):
        s, _ = env.reset(seed=int(rng.integers(1 << 30)))
        done = False
        while not done:
            # epsilon-greedy behavior policy
            if rng.random() < epsilon:
                a = int(rng.integers(n_a))
            else:
                a = int(np.argmax(q[s]))
            s_next, r, term, trun, _ = env.step(a)
            # Bellman backup; terminal states contribute no future value
            target = r if term else r + gamma * np.max(q[s_next])
            q[s, a] += alpha * (target - q[s, a])
            s = s_next
            done = term or trun
    return q


def greedy_policy(q):
    """Return the greedy action per state for a Q-table."""
    return np.argmax(q, axis=1)
