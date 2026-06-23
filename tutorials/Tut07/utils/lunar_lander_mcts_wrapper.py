"""MCTS wrapper for Gymnasium LunarLander.

`gymcts.gymcts_deepcopy_wrapper.DeepCopyMCTSGymEnvWrapper` cannot be used with
LunarLander because Box2D bodies do not survive `copy.deepcopy()` correctly. This
wrapper stores a state as `(reset seed, action history)` instead. Loading a state
resets the environment with the same seed and replays the actions.

This is slower than a true state serializer, but it is simple and works for the
standard deterministic LunarLander-v3 setup.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Callable, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType
from gymnasium.wrappers import RecordEpisodeStatistics

from gymcts.gymcts_env_abc import GymctsABC


@dataclass(frozen=True)
class LunarLanderMCTSState:
    """Serializable-ish state used by `LunarLanderMCTSWrapper`."""

    seed: int
    options: dict[str, Any] | None
    actions: tuple[int, ...]


class LunarLanderMCTSWrapper(GymctsABC, gym.Wrapper):
    """A GymCTS-compatible wrapper for discrete LunarLander-v3.

    The state is represented by the seed used at reset time plus the full action
    history since that reset. `load_state()` restores the state by resetting with
    that seed and replaying the actions.

    Important: this assumes deterministic dynamics from a fixed seed and action
    sequence. That is true for the default discrete LunarLander-v3 settings.
    """

    _terminal_flag: bool
    _step_tuple: tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]] | None
    _action_mask_fn: Callable[[gym.Env], np.ndarray] | None

    def __init__(
        self,
        env: gym.Env,
        action_mask_fn: str | Callable[[gym.Env], np.ndarray] | None = None,
        buffer_length: int = 100,
    ):
        env = RecordEpisodeStatistics(env, buffer_length=buffer_length)
        gym.Wrapper.__init__(self, env)

        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("Only discrete LunarLander action spaces are supported.")

        self._action_history: list[int] = []
        self._reset_seed: int | None = None
        self._reset_options: dict[str, Any] | None = None
        self._terminal_flag = False
        self._step_tuple = None
        self._action_mask_fn = None

        if action_mask_fn is not None:
            if isinstance(action_mask_fn, str):
                found_method = getattr(self.env, action_mask_fn)
                if not callable(found_method):
                    raise ValueError(f"Environment attribute {action_mask_fn} is not callable")
                self._action_mask_fn = found_method
            else:
                self._action_mask_fn = action_mask_fn

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)

        # For this replay-based wrapper, states must be reproducible from a seed.
        # Gymnasium exposes the actual generated seed here when `seed` was None
        # on a fresh env, but using an explicit seed is still recommended.
        self._reset_seed = int(self.unwrapped.np_random_seed)
        self._reset_options = copy.deepcopy(options)
        self._action_history = []
        self._terminal_flag = False
        self._step_tuple = None
        return obs, info

    def get_state(self) -> LunarLanderMCTSState:
        if self._reset_seed is None:
            raise RuntimeError("Call reset(seed=...) on the wrapper before using it with GymctsAgent.")

        return LunarLanderMCTSState(
            seed=self._reset_seed,
            options=copy.deepcopy(self._reset_options),
            actions=tuple(self._action_history),
        )

    def load_state(self, state: LunarLanderMCTSState) -> None:
        self.env.reset(seed=state.seed, options=copy.deepcopy(state.options))
        self._reset_seed = state.seed
        self._reset_options = copy.deepcopy(state.options)
        self._action_history = []
        self._terminal_flag = False
        self._step_tuple = None

        for action in state.actions:
            step_tuple = self.env.step(action)
            self._action_history.append(int(action))
            self._step_tuple = step_tuple
            _obs, _reward, terminated, truncated, _info = step_tuple
            self._terminal_flag = terminated or truncated

    def is_terminal(self) -> bool:
        return self._terminal_flag

    def action_masks(self) -> np.ndarray | None:
        return self._action_mask_fn(self.env) if self._action_mask_fn is not None else None

    def get_valid_actions(self) -> list[int]:
        if self._action_mask_fn is None:
            action_space: gym.spaces.Discrete = self.env.action_space
            return list(range(action_space.n))

        return [i for i, mask in enumerate(self.action_masks()) if mask]

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        step_tuple = self.env.step(action)
        self._action_history.append(int(action))

        _obs, _reward, terminated, truncated, _info = step_tuple
        self._terminal_flag = terminated or truncated
        self._step_tuple = step_tuple
        return step_tuple

    def rollout(self) -> float:
        is_terminal_state = self.is_terminal()

        if is_terminal_state:
            if self._step_tuple is None:
                raise RuntimeError("Terminal state has no recorded step tuple.")
            _obs, _reward, _terminated, _truncated, info = self._step_tuple
            return float(info["episode"]["r"])

        while not is_terminal_state:
            action = random.choice(self.get_valid_actions())
            _obs, _reward, is_terminal_state, _truncated, info = self.step(action)

        return float(info["episode"]["r"])
