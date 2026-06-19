"""Policy helpers for HW2 explorers.

This module provides two small containers used by the explorer agents:

- :class:`Policy` maps a state key (any hashable, typically a string built by
  a state-key function) to either a single action or an action distribution.
- :class:`QTable` stores tabular state-action values ``Q(s, a)`` and provides
  the basic operations needed by tabular Q-learning.

Neither container talks to the environment directly. The explorer is
responsible for translating observations into state keys before reading from
or writing to these containers.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np


class Policy:
    """Map state keys to actions or action distributions.

    A policy can be deterministic (one action per state) or stochastic (a
    distribution over actions per state). Both forms are stored side-by-side
    in :attr:`actions`: deterministic entries map a key to a single action
    string, while stochastic entries map a key to a ``{action: probability}``
    dict.

    Args:
        stochastic (bool): When ``True``, :meth:`get_action` samples from a
            stored action distribution; when ``False``, it returns the action
            with the highest stored probability.
        rng (Optional[np.random.Generator]): NumPy random generator used for
            sampling. Defaults to a fresh ``np.random.default_rng()``.
    """

    def __init__(self, stochastic: bool = False, rng=None):
        self.actions: Dict = {}
        self.stochastic = bool(stochastic)
        self.rng = rng if rng is not None else np.random.default_rng()

    def set_action(self, key, action) -> None:
        """Set the action (or action distribution) for a state key.

        Args:
            key: Hashable state key (typically the string returned by a
                ``state_key_fn``).
            action: Either an action name (str) for a deterministic entry, or
                a ``{action: probability}`` mapping for a stochastic entry.
                Mappings are normalized via :meth:`set_action_distribution`.
        """
        if isinstance(action, Mapping):
            self.set_action_distribution(key, action)
            return
        self.actions[key] = action

    def set_action_distribution(
        self,
        key,
        action_probabilities: Mapping[str, float],
    ) -> None:
        """Set a normalized stochastic action distribution for a state key.

        Args:
            key: Hashable state key.
            action_probabilities (Mapping[str, float]): Non-negative weights
                per action. Weights are renormalized to sum to 1.

        Raises:
            ValueError: If the mapping is empty, contains a negative weight,
                or has total weight ``<= 0``.
        """
        if not action_probabilities:
            raise ValueError("action_probabilities must not be empty")

        actions = []
        probabilities = []
        for action, probability in action_probabilities.items():
            probability = float(probability)
            if probability < 0:
                raise ValueError("action probabilities must be non-negative")
            actions.append(action)
            probabilities.append(probability)

        total_probability = sum(probabilities)
        if total_probability <= 0:
            raise ValueError("at least one action probability must be positive")

        normalized_probabilities = tuple(
            probability / total_probability
            for probability in probabilities
        )
        self.actions[key] = {
            action: probability
            for action, probability in zip(actions, normalized_probabilities)
        }

    def get_action(self, key):
        """Return the action stored for ``key`` (or ``None`` if unset).

        For stochastic entries the behavior depends on :attr:`stochastic`:

        - ``True``: sample an action from the distribution.
        - ``False``: return the action with the highest stored probability.

        Args:
            key: Hashable state key.

        Returns:
            Optional[str]: Action name, or ``None`` when ``key`` is not set.
        """
        action = self.actions.get(key)
        if action is None:
            return None
        if isinstance(action, Mapping):
            return self._sample_action(action) if self.stochastic else self._best_action(action)
        return action

    def get_action_distribution(self, key) -> Optional[Dict[str, float]]:
        """Return a copy of the action distribution for ``key``.

        Deterministic entries are returned as a single-entry dict with
        probability 1.0. Returns ``None`` when ``key`` is not present.
        """
        action = self.actions.get(key)
        if isinstance(action, Mapping):
            return dict(action)
        if action is None:
            return None
        return {action: 1.0}

    def set_stochastic(self, stochastic: bool) -> None:
        """Toggle stochastic vs. greedy lookup for distribution entries."""
        self.stochastic = bool(stochastic)

    def _sample_action(self, action_probabilities: Mapping[str, float]) -> str:
        actions = tuple(action_probabilities.keys())
        probabilities = tuple(action_probabilities.values())
        return self.rng.choice(actions, p=probabilities)

    def _best_action(self, action_probabilities: Mapping[str, float]) -> str:
        return max(action_probabilities, key=action_probabilities.get)

    def keys(self):
        """Iterate over stored state keys."""
        return self.actions.keys()

    def __contains__(self, key) -> bool:
        return key in self.actions

    def __getitem__(self, key) -> str:
        return self.actions[key]

    def __setitem__(self, key, action) -> None:
        self.set_action(key, action)


class QTable:
    """Tabular store for state-action values used by value-based RL explorers.

    Keys are ``(state_key, action)`` pairs; values are floats. Unset entries
    return :attr:`default_value`.

    Args:
        default_value (float): Value returned for ``(state_key, action)``
            pairs that have not been written. Defaults to ``0.0``.
    """

    def __init__(self, default_value: float = 0.0):
        self.default_value = float(default_value)
        self.values: Dict[Tuple[str, str], float] = {}

    def get_value(self, state_key: str, action: str) -> float:
        """Return ``Q(state_key, action)`` or :attr:`default_value`."""
        return self.values.get((state_key, action), self.default_value)

    def set_value(self, state_key: str, action: str, value: float) -> None:
        """Overwrite ``Q(state_key, action)`` with ``value``."""
        self.values[(state_key, action)] = float(value)

    def update_value(
        self,
        state_key: str,
        action: str,
        target: float,
        learning_rate: float,
    ) -> float:
        """Apply an in-place TD-style update.

        Computes ``Q <- Q + learning_rate * (target - Q)``.

        Args:
            state_key (str): State key.
            action (str): Action name.
            target (float): Target value (e.g., TD target).
            learning_rate (float): Step size, typically in ``(0, 1]``.

        Returns:
            float: The updated Q value.
        """
        old_value = self.get_value(state_key, action)
        new_value = old_value + learning_rate * (target - old_value)
        self.set_value(state_key, action, new_value)
        return new_value

    def get_best_action(self, state_key: str, available_actions) -> str:
        """Return the action in ``available_actions`` with the highest Q value.

        Ties are broken uniformly at random.

        Args:
            state_key (str): State key.
            available_actions (Iterable[str]): Actions to consider.

        Returns:
            str: Best (or randomly-chosen-among-best) action.
        """
        action_values = [
            self.get_value(state_key, action)
            for action in available_actions
        ]
        best_value = max(action_values)
        best_actions = [
            action
            for action, value in zip(available_actions, action_values)
            if value == best_value
        ]
        return np.random.choice(best_actions)

    def get_best_value(self, state_key: str, available_actions) -> float:
        """Return ``max_a Q(state_key, a)`` over ``available_actions``."""
        return max(
            self.get_value(state_key, action)
            for action in available_actions
        )

    def print_content(self) -> None:
        """Print all stored state-action values (sorted, for debugging)."""
        if not self.values:
            print("QTable is empty")
            return

        for (state_key, action), value in sorted(self.values.items()):
            print(f"state={state_key}, action={action}, value={value:.2f}")

    def __contains__(self, key) -> bool:
        return key in self.values

    def __getitem__(self, key) -> float:
        return self.values[key]

    def __setitem__(self, key, value: float) -> None:
        self.values[key] = float(value)
