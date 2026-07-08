"""Seeding and result-caching helpers shared across the Tut08 notebook."""

import pickle
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    """Seed torch, numpy, and the stdlib RNG for reproducibility.

    Args:
        seed: integer seed applied to all three generators.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_or_run(path, compute, desc=None):
    """Load a pickled result from ``path`` if it exists, else compute and cache it.

    This is the same load-or-run pattern used inline in Tutorial 07; it is
    factored out here so the notebook cells stay focused on the experiment
    definitions rather than the file handling.

    Args:
        path: str or Path of the pickle file to read/write.
        compute: zero-argument callable that produces the result when the
            cache is missing. Only called on a cache miss.
        desc: optional human-readable name used in the printed message.

    Returns:
        The cached or freshly computed object.
    """
    path = Path(path)
    name = desc or path.name
    if path.exists():
        with open(path, "rb") as f:
            result = pickle.load(f)
        print(f"Loaded cached {name} from {path}")
        return result
    # cache miss: run the experiment, then persist it for future sessions
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved {name} to {path}")
    return result
