"""Base agent interface for HW2 environments.

This module defines the :class:`Agent` base class and the canonical schemas for
``step_info`` (per-transition data passed to :meth:`Agent.step_update`) and
``episode_info`` (whole-episode data passed to :meth:`Agent.episode_update`).
Both schemas are stable across all explorer subclasses so that users can
rely on the same keys regardless of which decision-making approach they are
implementing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class Agent(ABC):
    """Base class for agents that act in an environment and (optionally) learn.

    Subclasses must implement:
    - :meth:`select_action`: choose the next action from an observation,
    - :meth:`step_update`: react to a single transition (e.g., TD update),
    - :meth:`episode_update`: react to a full trajectory (e.g., MC update).

    Training / evaluation mode is controlled by ``self.training``:
    - ``training=True`` (default): subclasses should apply learning updates from
      inside :meth:`step_update` / :meth:`episode_update`.
    - ``training=False``: subclasses should still update internal bookkeeping
      (e.g., best value observed so far) but skip learning updates.

    Use :meth:`train` and :meth:`eval` to toggle this flag.

    The base class deliberately does not provide or require a ``reset()`` hook.
    Any per-episode state lives either on the explorer subclass or in the global
    aggregators in `to_implement.py`. The runner never silently
    resets agent state across rollouts -- if you need to clear something
    between episodes, do it explicitly.

    Attributes:
        env: The Gym-style environment the agent acts in by default.
        training (bool): Whether learning updates should be applied.
    """

    def __init__(self, env):
        """Initialize the base agent.

        Args:
            env: A Gym-style environment. The agent will act in this
                environment by default when :meth:`run_episode` is called.
        """
        self.env = env
        self.training: bool = True

    def train(self) -> None:
        """Enable learning updates for subsequent rollouts."""
        self.training = True

    def eval(self) -> None:
        """Disable learning updates for subsequent rollouts."""
        self.training = False

    @abstractmethod
    def select_action(self, obs: dict, info: dict) -> str:
        """Choose the next action given the current observation.

        Args:
            obs (dict): Observation returned by ``env.reset`` at the start of
                the episode, or by the previous ``env.step`` call.
            info (dict): Auxiliary ``info`` dict returned alongside ``obs``.

        Returns:
            str: Action name (one of :data:`caldera_env.ACTION_NAMES`).
        """

    @abstractmethod
    def step_update(self, step_info: dict) -> None:
        """Apply a per-transition update after the environment has been stepped.

        The runner constructs ``step_info`` as a dictionary with the following
        keys (these keys are guaranteed and stable across subclasses):

        - ``"obs"`` (dict): observation BEFORE the action was taken.
        - ``"action"`` (str): action that was actually executed.
        - ``"reward"`` (float): scalar reward received from the environment.
        - ``"next_obs"`` (dict): observation AFTER the action.
        - ``"terminated"`` (bool): episode ended due to a terminal condition
          (e.g., energy depleted, collision when
          ``end_episode_on_collision``).
        - ``"truncated"`` (bool): episode was cut short externally.
        - ``"info"`` (dict): auxiliary ``info`` dict returned by ``env.step``
          alongside ``next_obs`` (may contain ``"action_mask"`` etc.).

        Subclasses typically call ``super().update_best_value(step_info)`` and,
        when ``self.training`` is ``True``, dispatch to a user-provided
        ``step_update_fn`` for TD-style learning.

        Args:
            step_info (dict): Single-transition data as described above.
        """

    @abstractmethod
    def episode_update(self, episode_info: dict) -> None:
        """Apply an end-of-episode update after the episode terminates.

        The runner constructs ``episode_info`` as a dictionary with:

        - ``"trajectory"`` (list[dict]): all ``step_info`` dicts collected
          during the episode, in temporal order.
        - ``"total_reward"`` (float): sum of rewards over the episode.
        - ``"num_steps"`` (int): number of transitions in the episode.

        Subclasses typically dispatch to a user-provided ``episode_update_fn``
        for Monte-Carlo-style learning when ``self.training`` is ``True``.

        Args:
            episode_info (dict): Whole-episode data as described above.
        """

    def is_done(self) -> bool:
        """Return whether the agent itself wants to end the episode early.

        The default implementation never asks to terminate. Subclasses such as
        :class:`explorer.ALExplorer` override this to stop after a sample
        budget is exhausted.
        """
        return False

    def run_episode(self, env=None, render: bool = False) -> dict:
        """Run a single episode end-to-end.

        At every step the runner:

        1. asks the agent for an action via :meth:`select_action`,
        2. steps the environment,
        3. constructs ``step_info`` and calls :meth:`step_update`.

        After the episode ends (terminated, truncated, or ``is_done()``), the
        runner builds ``episode_info`` and calls :meth:`episode_update` exactly
        once.

        The runner does **not** auto-reset the agent between rollouts. Any
        per-episode state must be cleared explicitly by the agent's owner if
        the agent will be reused across episodes.

        Args:
            env: Optional environment override. Defaults to ``self.env``.
            render (bool): When ``True``, call ``env.render()`` and print a
                short summary line after every step.

        Returns:
            dict: ``episode_info`` as described in :meth:`episode_update`.
        """
        if env is None:
            env = self.env

        obs, info = env.reset()
        trajectory: List[dict] = []
        total_reward = 0.0
        env_terminated = False
        env_truncated = False

        while not (env_terminated or env_truncated or self.is_done()):
            action = self.select_action(obs, info)
            next_obs, reward, env_terminated, env_truncated, next_info = env.step(action)
            step_info = {
                "obs": obs,
                "action": action,
                "reward": float(reward),
                "next_obs": next_obs,
                "terminated": bool(env_terminated),
                "truncated": bool(env_truncated),
                "info": next_info,
            }
            self.step_update(step_info)
            trajectory.append(step_info)
            total_reward += float(reward)

            if render:
                env.render()
                print(
                    f"Step: {len(trajectory)}, "
                    f"position: {next_obs.get('position')}, "
                    f"Action: {action}, Reward: {reward:.2f}, "
                    f"Total Reward: {total_reward:.2f}"
                )

            obs = next_obs
            info = next_info

        episode_info = {
            "trajectory": trajectory,
            "total_reward": float(total_reward),
            "num_steps": len(trajectory),
        }
        self.episode_update(episode_info)
        return episode_info

    def run_episodes(
        self,
        env=None,
        num_episodes: int = 1,
        render: bool = False,
    ) -> List[dict]:
        """Run ``num_episodes`` consecutive episodes and return their summaries.

        Args:
            env: Optional environment override. Defaults to ``self.env``.
            num_episodes (int): Number of episodes to run. Must be non-negative.
            render (bool): Forwarded to :meth:`run_episode`.

        Returns:
            List[dict]: One ``{"episode": index, "episode_info": ...}`` entry
            per episode, in order.

        Raises:
            ValueError: If ``num_episodes`` is negative.
        """
        if env is None:
            env = self.env
        if num_episodes < 0:
            raise ValueError("num_episodes must be non-negative")

        episodes: List[dict] = []
        for episode_index in range(num_episodes):
            episode_info = self.run_episode(env=env, render=render)
            episodes.append({"episode": episode_index, "episode_info": episode_info})
        return episodes
