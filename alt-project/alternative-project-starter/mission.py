"""The Caldera Return Mission.

PARLAI (02360765), Spring 2026, alternative project.

This file builds your personal mission and the environment you will work with.
Do not change it. We grade your code with our own copy of this file.

Main entry points
-----------------
make_mission(student_id)   -> (env, info)
evaluate(policy, env, n_episodes, first_seed=0)
run_episode(policy, env, seed)
plot_mission(info), plot_path(info, episode), plot_arrows(info, actions_by_cell)
describe_mission(info), depth_at(info, x, y)

A policy is any callable policy(obs, info) that returns an action, either as a
name from ACTIONS or as its index. If the policy object has a reset() method,
run_episode and evaluate call it at the start of every episode.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Mission constants
# ---------------------------------------------------------------------------

GRID_SIZE = 9
NUM_TARGETS = 3
NUM_OBSTACLES = 7

MOVE_NORTH = "MOVE_NORTH"
MOVE_SOUTH = "MOVE_SOUTH"
MOVE_EAST = "MOVE_EAST"
MOVE_WEST = "MOVE_WEST"
SAMPLE = "SAMPLE"
DOCK = "DOCK"
ACTIONS = (MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST, SAMPLE, DOCK)
ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}
MOVE_ACTIONS = (MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST)

# (dx, dy) of each move. North increases y, east increases x.
MOVE_DELTAS = {
    MOVE_NORTH: (0, 1),
    MOVE_SOUTH: (0, -1),
    MOVE_EAST: (1, 0),
    MOVE_WEST: (-1, 0),
}

# The move that actually happens when the current turns the vehicle.
RIGHT_TURN = {
    MOVE_NORTH: MOVE_EAST,
    MOVE_EAST: MOVE_SOUTH,
    MOVE_SOUTH: MOVE_WEST,
    MOVE_WEST: MOVE_NORTH,
}

STEP_REWARD = -1.0
TARGET_REWARD = 25.0
LOSS_PENALTY = -50.0
# Generator settings. They keep the missions of different students comparable.
ENERGY_MARGIN = 10
MIN_PLAN_COST = 24
MAX_PLAN_COST = 28
MAX_DETOUR = 2
P_BAR_RANGE = (0.80, 0.85)

Cell = Tuple[int, int]
State = Tuple[Cell, int, Tuple[int, ...]]


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

def _manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _in_map(c: Cell) -> bool:
    return 0 <= c[0] < GRID_SIZE and 0 <= c[1] < GRID_SIZE


def _neighbors(c: Cell) -> List[Cell]:
    return [(c[0] + dx, c[1] + dy) for dx, dy in MOVE_DELTAS.values()]


def _reachable(start: Cell, obstacles: set) -> set:
    seen = {start}
    queue = deque([start])
    while queue:
        c = queue.popleft()
        for n in _neighbors(c):
            if _in_map(n) and n not in obstacles and n not in seen:
                seen.add(n)
                queue.append(n)
    return seen


def _neighbors8(c: Cell) -> List[Cell]:
    return [(c[0] + dx, c[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]


def deterministic_plan_cost(dock: Cell, targets: Sequence[Cell], obstacles: set) -> Optional[int]:
    """Optimal number of actions to collect all targets and dock when p = 1.

    Counts moves, SAMPLE actions and the final DOCK. Returns None if impossible.
    """
    return _tour_cost(dock, targets, obstacles, avoid_turn_hazards=False)


def _tour_cost(dock: Cell, targets: Sequence[Cell], obstacles: set, avoid_turn_hazards: bool) -> Optional[int]:
    """Number of actions of the shortest tour that collects all targets and docks.

    With avoid_turn_hazards, the tour may only use moves for which a turn to the
    right cannot enter an obstacle. The generator uses this to keep missions
    comparable between students.
    """
    full = (1 << len(targets)) - 1
    start = (dock, 0)
    dist = {start: 0}
    queue = deque([start])
    while queue:
        cell, mask = queue.popleft()
        d = dist[(cell, mask)]
        if mask == full and cell == dock:
            return d + 1  # + DOCK
        successors = []
        for name, (dx, dy) in MOVE_DELTAS.items():
            n = (cell[0] + dx, cell[1] + dy)
            if not _in_map(n) or n in obstacles:
                continue
            if avoid_turn_hazards:
                rx, ry = MOVE_DELTAS[RIGHT_TURN[name]]
                if (cell[0] + rx, cell[1] + ry) in obstacles:
                    continue
            successors.append((n, mask))
        if cell in targets:
            i = targets.index(cell)
            successors.append((cell, mask | (1 << i)))
        for s in successors:
            if s not in dist:
                dist[s] = d + 1
                queue.append(s)
    return None


def _generate_layout(rng: np.random.Generator):
    corners = [(0, 0), (0, GRID_SIZE - 1), (GRID_SIZE - 1, 0), (GRID_SIZE - 1, GRID_SIZE - 1)]
    all_cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
    while True:
        dock = corners[int(rng.integers(len(corners)))]
        near_dock = set(_neighbors8(dock)) | {dock}
        candidates = [c for c in all_cells if c not in near_dock]
        idx = rng.choice(len(candidates), size=NUM_OBSTACLES + NUM_TARGETS, replace=False)
        obstacles = [candidates[i] for i in idx[:NUM_OBSTACLES]]
        targets = [candidates[i] for i in idx[NUM_OBSTACLES:]]
        points = targets + [dock]
        if any(_manhattan(points[i], points[j]) < 3
               for i in range(len(points)) for j in range(i + 1, len(points))):
            continue
        obstacle_set = set(obstacles)
        if any(n in obstacle_set for t in targets for n in _neighbors(t)):
            continue
        reach = _reachable(dock, obstacle_set)
        if not all(t in reach for t in targets):
            continue
        cost = deterministic_plan_cost(dock, targets, obstacle_set)
        if cost is None or not (MIN_PLAN_COST <= cost <= MAX_PLAN_COST):
            continue
        careful_cost = _tour_cost(dock, targets, obstacle_set, avoid_turn_hazards=True)
        if careful_cost is None or careful_cost > cost + MAX_DETOUR:
            continue
        # sort for a stable, readable order
        targets = sorted(targets)
        obstacles = sorted(obstacles)
        return dock, targets, obstacles, cost


def _generate_currents(rng: np.random.Generator):
    while True:
        a = round(float(rng.uniform(0.05, 0.15)), 2)
        b = round(float(rng.uniform(0.20, 0.40)), 2)
        p_calm = round(float(rng.uniform(0.90, 0.95)), 2)
        p_strong = round(float(rng.uniform(0.45, 0.60)), 2)
        pi_strong = a / (a + b)
        p_bar = (1 - pi_strong) * p_calm + pi_strong * p_strong
        if P_BAR_RANGE[0] <= p_bar <= P_BAR_RANGE[1]:
            return a, b, p_calm, p_strong, p_bar


def _generate_current_log(rng: np.random.Generator, a, b, p_calm, p_strong, length=60):
    pi_strong = a / (a + b)
    while True:
        regimes, outcomes = [], []
        r = "Strong" if rng.random() < pi_strong else "Calm"
        for t in range(length):
            if t > 0:
                if r == "Calm" and rng.random() < a:
                    r = "Strong"
                elif r == "Strong" and rng.random() < b:
                    r = "Calm"
            p = p_calm if r == "Calm" else p_strong
            regimes.append(r)
            outcomes.append(int(rng.random() < p))
        changes = sum(regimes[i] != regimes[i - 1] for i in range(1, length))
        if changes >= 2 and regimes.count("Strong") >= 10:
            return {"outcomes": outcomes, "regimes": regimes}


def _generate_pits(targets, rng):
    """One Gaussian pit at each target. The depth is used only for figures."""
    return [{"x": int(t[0]), "y": int(t[1]),
             "sigma": round(float(rng.uniform(0.9, 1.4)), 2),
             "weight": round(float(rng.uniform(120, 200)), 1)} for t in targets]


def depth_at(info: dict, x, y):
    """Depth in meters (negative) at map coordinates x, y. Used only for figures."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    c = (info["grid_size"] - 1) / 2
    total = 40.0 + 25.0 * (1 - ((x - c) ** 2 + (y - c) ** 2) / (2 * c * c))
    for pit in info["pits"]:
        s = pit["sigma"]
        total = total + pit["weight"] * np.exp(-((x - pit["x"]) ** 2 + (y - pit["y"]) ** 2) / (2 * s * s))
    return -total


def make_mission(student_id) -> Tuple["MissionEnv", dict]:
    """Build the mission of one student.

    student_id: your 9-digit ID number (int or str). The same ID always gives
    the same mission.
    Returns (env, info).
    """
    sid = int(str(student_id).strip())
    rng = np.random.default_rng(sid)
    dock, targets, obstacles, plan_cost = _generate_layout(rng)
    a, b, p_calm, p_strong, p_bar = _generate_currents(rng)
    log = _generate_current_log(rng, a, b, p_calm, p_strong)
    pits = _generate_pits(targets, rng)

    info = {
        "student_id": sid,
        "grid_size": GRID_SIZE,
        "dock": dock,
        "targets": targets,
        "obstacles": obstacles,
        "energy": plan_cost + ENERGY_MARGIN,
        "p": p_bar,
        "rewards": {"step": STEP_REWARD, "target": TARGET_REWARD, "lost": LOSS_PENALTY},
        "currents": {"a": a, "b": b, "p_calm": p_calm, "p_strong": p_strong},
        "current_log": log,
        "pits": pits,  # for figures only
    }
    env = MissionEnv(info)
    return env, info


def describe_mission(info: dict) -> str:
    """A short text description of the mission for the start of your report."""
    c = info["currents"]
    lines = [
        f"Student ID: {info['student_id']}",
        f"Grid: {info['grid_size']} x {info['grid_size']}",
        f"Dock: {info['dock']}",
        f"Targets: {info['targets']}",
        f"Obstacles: {info['obstacles']}",
        f"Energy E: {info['energy']}",
        f"Success probability p: {info['p']:.4f}",
        f"Currents: a={c['a']}, b={c['b']}, p_C={c['p_calm']}, p_S={c['p_strong']}",
        "Current log outcomes: " + "".join(str(o) for o in info["current_log"]["outcomes"]),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MissionEnv(gym.Env):
    """Gymnasium environment for the Caldera Return Mission.

    Observation: a dict with
        "cell":      np.array([x, y])
        "energy":    remaining energy (int)
        "collected": np.array of 3 zeros and ones, one per target in info["targets"]
    Info dict from reset() and step():
        "action_mask": np.array of 6 zeros and ones, in the order of ACTIONS
        "turned":      True if the last move was turned by the current
        "crashed", "lost", "docked": end-of-episode flags
    Calling step() with an action that is not available raises ValueError.
    """

    metadata = {"render_modes": []}

    def __init__(self, info: dict):
        super().__init__()
        self.mission = info
        self.dock: Cell = tuple(info["dock"])
        self.targets: List[Cell] = [tuple(t) for t in info["targets"]]
        self.obstacles = set(tuple(o) for o in info["obstacles"])
        self.initial_energy = int(info["energy"])
        self.p = float(info["p"])
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Dict({
            "cell": spaces.MultiDiscrete([GRID_SIZE, GRID_SIZE]),
            "energy": spaces.Discrete(self.initial_energy + 1),
            "collected": spaces.MultiBinary(NUM_TARGETS),
        })
        self._set(self.dock, self.initial_energy, (0,) * NUM_TARGETS)

    # -- state handling ----------------------------------------------------
    def _set(self, cell, energy, collected):
        self.cell = (int(cell[0]), int(cell[1]))
        self.energy = int(energy)
        self.collected = [int(c) for c in collected]
        self.done = False
        self.crashed = False
        self.lost = False
        self.docked = False

    def set_state(self, cell: Cell, energy: int, collected: Sequence[int]) -> Tuple[dict, dict]:
        """Put the environment in any state. collected has one 0/1 value per target.

        Allowed only where the project rules allow it.
        Returns (obs, info) for the new state.
        """
        cell = (int(cell[0]), int(cell[1]))
        if not _in_map(cell) or cell in self.obstacles:
            raise ValueError(f"{cell} is not a free cell")
        if not 1 <= int(energy) <= self.initial_energy:
            raise ValueError("energy must be between 1 and E")
        if len(collected) != NUM_TARGETS:
            raise ValueError("collected must have one value per target")
        self._set(cell, energy, collected)
        return self._obs(), self._info(turned=False)

    def get_state(self) -> State:
        """The current state as a hashable tuple (cell, energy, collected)."""
        return self.cell, self.energy, tuple(self.collected)

    def available_actions(self, state: Optional[State] = None) -> List[str]:
        """Names of the actions the vehicle may choose in a state (default: current state)."""
        if state is None:
            cell, energy, collected = self.get_state()
        else:
            cell, energy, collected = state
        if energy <= 0:
            return []
        acts = []
        for name in MOVE_ACTIONS:
            dx, dy = MOVE_DELTAS[name]
            n = (cell[0] + dx, cell[1] + dy)
            if _in_map(n) and n not in self.obstacles:
                acts.append(name)
        acts.append(SAMPLE)
        if tuple(cell) == self.dock and sum(collected) >= 1:
            acts.append(DOCK)
        return acts

    def action_mask(self, state: Optional[State] = None) -> np.ndarray:
        avail = set(self.available_actions(state))
        return np.array([1 if a in avail else 0 for a in ACTIONS], dtype=np.int8)

    def _obs(self) -> dict:
        return {
            "cell": np.array(self.cell, dtype=np.int64),
            "energy": self.energy,
            "collected": np.array(self.collected, dtype=np.int8),
        }

    def _info(self, turned: bool) -> dict:
        return {
            "action_mask": self.action_mask() if not self.done else np.zeros(len(ACTIONS), dtype=np.int8),
            "turned": turned,
            "crashed": self.crashed,
            "lost": self.lost,
            "docked": self.docked,
        }

    # -- gymnasium API -------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._set(self.dock, self.initial_energy, (0,) * NUM_TARGETS)
        return self._obs(), self._info(turned=False)

    def step(self, action):
        if self.done:
            raise RuntimeError("The episode has ended. Call reset() or set_state().")
        name = ACTIONS[int(action)] if not isinstance(action, str) else action
        if name not in self.available_actions():
            raise ValueError(f"Action {name} is not available in state {self.get_state()}")

        reward = STEP_REWARD
        turned = False
        self.energy -= 1

        if name in MOVE_ACTIONS:
            executed = name
            if self.np_random.random() >= self.p:
                executed = RIGHT_TURN[name]
                turned = True
            dx, dy = MOVE_DELTAS[executed]
            n = (self.cell[0] + dx, self.cell[1] + dy)
            if not _in_map(n):
                pass  # the vehicle stays in its cell
            elif n in self.obstacles:
                self.crashed = True
                self.lost = True
                self.done = True
                reward += LOSS_PENALTY
            else:
                self.cell = n
        elif name == SAMPLE:
            if self.cell in self.targets:
                i = self.targets.index(self.cell)
                if not self.collected[i]:
                    self.collected[i] = 1
                    reward += TARGET_REWARD
        elif name == DOCK:
            self.docked = True
            self.done = True

        if not self.done and self.energy == 0:
            self.done = True
            if self.cell != self.dock:
                self.lost = True
                reward += LOSS_PENALTY

        return self._obs(), reward, self.done, False, self._info(turned)


def obs_to_state(obs: dict) -> State:
    """Convert an observation to the hashable state (cell, energy, collected)."""
    return (int(obs["cell"][0]), int(obs["cell"][1])), int(obs["energy"]), tuple(int(c) for c in obs["collected"])


# ---------------------------------------------------------------------------
# Running and evaluating policies
# ---------------------------------------------------------------------------

def run_episode(policy: Callable, env: MissionEnv, seed: Optional[int] = None) -> dict:
    """Run one episode and return a record of it (used by plot_path)."""
    if hasattr(policy, "reset"):
        policy.reset()
    obs, info = env.reset(seed=seed)
    cells = [tuple(env.cell)]
    actions, turned, rewards = [], [], []
    done = False
    while not done:
        action = policy(obs, info)
        name = ACTIONS[int(action)] if not isinstance(action, str) else action
        obs, reward, done, _, info = env.step(name)
        actions.append(name)
        turned.append(bool(info["turned"]))
        rewards.append(float(reward))
        cells.append(tuple(env.cell))
    collected = tuple(env.collected)
    return {
        "seed": seed,
        "cells": cells,
        "actions": actions,
        "turned": turned,
        "rewards": rewards,
        "return": float(sum(rewards)),
        "collected": collected,
        "crashed": env.crashed,
        "lost": env.lost,
        "docked": env.docked,
        "full_success": (sum(collected) == NUM_TARGETS) and not env.lost,
    }


def evaluate(policy: Callable, env: MissionEnv, n_episodes: int, first_seed: int = 0,
             keep_episodes: bool = False) -> dict:
    """Run n_episodes episodes with seeds first_seed, first_seed+1, ...

    Returns a dict of numpy arrays: "returns", "full_success", "lost", "crashed",
    and the summary numbers "mean_return", "std_error", "full_success_rate", "loss_rate".
    With keep_episodes=True it also returns the episode records under "episodes".
    """
    episodes = [run_episode(policy, env, seed=first_seed + i) for i in range(n_episodes)]
    returns = np.array([e["return"] for e in episodes])
    result = {
        "returns": returns,
        "full_success": np.array([e["full_success"] for e in episodes]),
        "lost": np.array([e["lost"] for e in episodes]),
        "crashed": np.array([e["crashed"] for e in episodes]),
        "mean_return": float(returns.mean()),
        "std_error": float(returns.std(ddof=1) / np.sqrt(n_episodes)) if n_episodes > 1 else float("nan"),
        "full_success_rate": float(np.mean([e["full_success"] for e in episodes])),
        "loss_rate": float(np.mean([e["lost"] for e in episodes])),
    }
    if keep_episodes:
        result["episodes"] = episodes
    return result


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_mission(info: dict, ax=None, show_depth: bool = True, title: Optional[str] = None):
    """Draw the map with the depth field, obstacles, dock and targets. Returns (fig, ax)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
    else:
        fig = ax.figure
    n = info["grid_size"]
    if show_depth:
        xs = np.linspace(-0.5, n - 0.5, 200)
        X, Y = np.meshgrid(xs, xs)
        cs = ax.contourf(X, Y, depth_at(info, X, Y), levels=20, cmap="viridis")
        fig.colorbar(cs, ax=ax, label="depth (m)", fraction=0.046, pad=0.04)
    for k in range(n + 1):
        ax.axhline(k - 0.5, color="white", lw=0.6, alpha=0.6)
        ax.axvline(k - 0.5, color="white", lw=0.6, alpha=0.6)
    for (x, y) in info["obstacles"]:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="black", edgecolor="white", lw=1, zorder=3))
    dx, dy = info["dock"]
    ax.add_patch(Rectangle((dx - 0.5, dy - 0.5), 1, 1, facecolor="#f2a900", edgecolor="white", lw=1.5, zorder=3))
    ax.text(dx, dy, "Dock", ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)
    for i, (x, y) in enumerate(info["targets"]):
        ax.scatter([x], [y], s=380, c="#d62728", edgecolors="white", linewidths=1.5, zorder=4)
        ax.text(x, y, f"T{i}", ha="center", va="center", color="white", fontsize=8, fontweight="bold", zorder=5)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("x (east)")
    ax.set_ylabel("y (north)")
    ax.set_aspect("equal")
    ax.set_title(title if title is not None else f"Mission of {info['student_id']}")
    return fig, ax


def plot_path(info: dict, episode: dict, ax=None, title: Optional[str] = None):
    """Draw one episode record from run_episode on the map. Returns (fig, ax).

    Blue segments are moves in the intended direction. Orange dashed segments are
    moves turned by the current. White stars mark collected targets. A red X marks
    a crash and the obstacle it hit.
    """
    import matplotlib.pyplot as plt

    fig, ax = plot_mission(info, ax=ax, show_depth=True, title=title or "")
    cells = episode["cells"]
    collected_cells = set()
    for k, action in enumerate(episode["actions"]):
        a, b = cells[k], cells[k + 1]
        if action in MOVE_ACTIONS:
            executed = RIGHT_TURN[action] if episode["turned"][k] else action
            if a != b:
                style = dict(color="#ff7f0e", ls="--", lw=2.5) if episode["turned"][k] else dict(color="#1f77b4", lw=2.5)
                ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", shrinkA=6, shrinkB=6, **style), zorder=6)
            if k == len(episode["actions"]) - 1 and episode["crashed"]:
                ddx, ddy = MOVE_DELTAS[executed]
                hit = (a[0] + ddx, a[1] + ddy)
                ax.scatter([hit[0]], [hit[1]], marker="x", s=220, c="red", linewidths=3, zorder=7)
        elif action == SAMPLE and episode["rewards"][k] > 0:
            collected_cells.add(a)
    for (x, y) in collected_cells:
        ax.scatter([x], [y], marker="*", s=260, c="white", edgecolors="black", zorder=7)
    if title is None:
        status = "docked" if episode["docked"] else ("crashed" if episode["crashed"] else ("lost" if episode["lost"] else "ended"))
        ax.set_title(f"Return {episode['return']:.0f}, {sum(episode['collected'])} of {NUM_TARGETS} targets, {status}")
    return fig, ax


def plot_arrows(info: dict, actions_by_cell: Dict[Cell, str], ax=None, title: Optional[str] = None):
    """Draw one action per cell as an arrow (moves), a dot (SAMPLE) or a D (DOCK)."""
    fig, ax = plot_mission(info, ax=ax, show_depth=True, title=title or "")
    for (x, y), action in actions_by_cell.items():
        if action in MOVE_ACTIONS:
            dx, dy = MOVE_DELTAS[action]
            ax.arrow(x - 0.25 * dx, y - 0.25 * dy, 0.4 * dx, 0.4 * dy, head_width=0.18, head_length=0.12,
                     length_includes_head=True, color="white", zorder=6)
        elif action == SAMPLE:
            ax.scatter([x], [y], s=40, c="white", zorder=6)
        elif action == DOCK:
            ax.text(x, y - 0.3, "D", color="white", ha="center", va="center", fontsize=9, fontweight="bold", zorder=6)
    return fig, ax


if __name__ == "__main__":
    import sys

    sid = sys.argv[1] if len(sys.argv) > 1 else "123456789"
    _, mission_info = make_mission(sid)
    print(describe_mission(mission_info))
