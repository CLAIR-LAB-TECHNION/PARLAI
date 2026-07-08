"""Plotting helpers for the Tut08 notebook.

Everything here is presentation-only: learning-curve smoothing, the VaultEnv
objective-landscape figures, the gradient-noise visualizations, and the AimEnv
policy plots. No algorithmic content lives in this module.
"""

import matplotlib.pyplot as plt
import numpy as np


def rolling_mean(x, window=20):
    """Trailing rolling mean (same helper as Tut07).

    Args:
        x: 1-D sequence of values.
        window: number of trailing samples to average; 1 disables smoothing.

    Returns:
        numpy array of the same length as ``x``.
    """
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out[i] = x[lo:i + 1].mean()
    return out


def plot_seed_curves(runs, window=20, color="C0", label=None, ax=None,
                     n_points=300, seed_alpha=0.3):
    """Plot per-seed learning curves (faint) and their mean (bold).

    Each run is a list of ``(x, y)`` pairs, e.g. ``(episode, return)`` or
    ``(env_steps, return)``. Runs whose x-coordinates differ (as happens when
    x is an environment-step count) are linearly interpolated onto a common
    grid before averaging.

    Args:
        runs: list over seeds of lists of (x, y) tuples.
        window: rolling-mean window applied to each seed's y-values.
        color: matplotlib color shared by the seed curves and the mean.
        label: legend label attached to the mean curve.
        ax: axes to draw on (defaults to current axes).
        n_points: resolution of the common x-grid used for the mean.
        seed_alpha: opacity of the individual seed curves.

    Returns:
        (grid, mean) arrays of the common x-grid and the mean curve on it.
    """
    ax = ax or plt.gca()
    xs = [np.asarray([p[0] for p in run], dtype=float) for run in runs]
    ys = [rolling_mean([p[1] for p in run], window) for run in runs]
    # common grid: from the largest start to the smallest end so every run
    # covers the full grid and the mean is never extrapolated
    lo = max(x[0] for x in xs)
    hi = min(x[-1] for x in xs)
    grid = np.linspace(lo, hi, n_points)
    interp = np.stack([np.interp(grid, x, y) for x, y in zip(xs, ys)])
    for x, y in zip(xs, ys):
        ax.plot(x, y, color=color, alpha=seed_alpha)
    mean = interp.mean(axis=0)
    ax.plot(grid, mean, color=color, linewidth=2.5, label=label)
    return grid, mean


def plot_vault_landscape(J, lim=4.0, ax=None, grad=None, n_grid=200,
                         n_quiver=13, annotate=True, cbar=True):
    """Draw the exact objective surface of VaultEnv over parameter space.

    Args:
        J: vectorized function J(theta0, theta1) -> expected return; called on
            meshgrid arrays.
        lim: the plot covers [-lim, lim]^2 in both parameters.
        ax: axes to draw on (defaults to current axes).
        grad: optional function (theta0, theta1) -> (dJ/dtheta0, dJ/dtheta1);
            when given, the true gradient field is overlaid as a quiver plot.
        n_grid: contour resolution.
        n_quiver: quiver arrows per axis.
        annotate: mark the global optimum and the decoy attractor regions.
        cbar: attach a colorbar.

    Returns:
        The contour set (useful for adding a shared colorbar).
    """
    ax = ax or plt.gca()
    t = np.linspace(-lim, lim, n_grid)
    T0, T1 = np.meshgrid(t, t)
    Z = J(T0, T1)
    cs = ax.contourf(T0, T1, Z, levels=30, cmap="viridis")
    if cbar:
        plt.colorbar(cs, ax=ax, label=r"$J(\theta)$")
    if grad is not None:
        q = np.linspace(-lim, lim, n_quiver)
        Q0, Q1 = np.meshgrid(q, q)
        G0, G1 = grad(Q0, Q1)
        ax.quiver(Q0, Q1, G0, G1, color="white", alpha=0.8, width=0.004)
    if annotate:
        # fixed positions so the labels stay in the interesting region even
        # when the plot is drawn with a larger lim
        ax.annotate("global optimum\n(enter, right lever)", xy=(2.8, 2.9),
                    ha="center", color="white", fontsize=9, fontweight="bold")
        ax.annotate("decoy attractor\n(always take the exit)", xy=(-2.4, -2.2),
                    ha="center", color="white", fontsize=9, fontweight="bold")
        # basin boundary: entering pays off iff sigma(theta1) > exit reward
        ax.axhline(0.0, color="red", linestyle="--", alpha=0.6, linewidth=1)
    ax.set_xlabel(r"$\theta_0$  (enter the vault $\leftrightarrow$ take the exit)")
    ax.set_ylabel(r"$\theta_1$  (right lever $\leftrightarrow$ wrong lever)")
    return cs


def plot_trajectories(trajs, ax, color="C1", endpoint_color=None, lw=1.5, alpha=0.9):
    """Overlay parameter-space trajectories on a landscape plot.

    Args:
        trajs: list of arrays of shape (T, 2), each one optimization run.
        ax: axes holding the landscape.
        color: line color.
        endpoint_color: color of the endpoint marker (defaults to ``color``).
        lw, alpha: line width and opacity.
    """
    endpoint_color = endpoint_color or color
    for tr in trajs:
        tr = np.asarray(tr)
        ax.plot(tr[:, 0], tr[:, 1], color=color, linewidth=lw, alpha=alpha)
        ax.plot(tr[0, 0], tr[0, 1], "o", color=color, markersize=4)          # start
        ax.plot(tr[-1, 0], tr[-1, 1], "*", color=endpoint_color, markersize=12,
                markeredgecolor="k", markeredgewidth=0.5)                     # end


def plot_gradient_samples(samples, true_grad, ax=None, color="C0", label=None,
                          lim=None, alpha=0.35):
    """Scatter sampled 2-D gradient estimates around the true gradient.

    Args:
        samples: array (N, 2) of gradient estimates at a fixed theta.
        true_grad: length-2 array, the exact gradient at that theta.
        ax: axes to draw on (defaults to current axes).
        color: scatter color.
        label: legend label for the scatter cloud.
        lim: symmetric axis limit; computed from the samples when None.
        alpha: scatter opacity.
    """
    ax = ax or plt.gca()
    samples = np.asarray(samples)
    mean = samples.mean(axis=0)
    ax.scatter(samples[:, 0], samples[:, 1], s=12, alpha=alpha, color=color, label=label)
    # true gradient (black) and empirical mean of the estimates (colored)
    ax.annotate("", xy=tuple(true_grad), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="k", lw=2.5))
    ax.annotate("", xy=tuple(mean), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=color, lw=2))
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    if lim is None:
        lim = float(np.abs(samples).max()) * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$\hat g_{\theta_0}$")
    ax.set_ylabel(r"$\hat g_{\theta_1}$")
    ax.set_aspect("equal")


def plot_probe_histograms(projections, bins=40):
    """Compare gradient-estimate histograms across estimator variants.

    Draws one panel per variant with the histogram of the uphill components of
    the single-episode gradient estimates (their projections onto the
    reference/mean direction). Mass below zero corresponds to estimates whose
    update step would move the policy downhill.

    Args:
        projections: dict mapping variant name -> 1-D array of projections.
        bins: histogram bin count (shared range across panels).
    """
    names = list(projections)
    all_vals = np.concatenate([np.asarray(v) for v in projections.values()])
    rng = (all_vals.min(), all_vals.max())
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 3.6),
                             sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, name, color in zip(axes, names, [f"C{i}" for i in range(len(names))]):
        vals = np.asarray(projections[name])
        frac_neg = (vals < 0).mean()
        ax.hist(vals, bins=bins, range=rng, color=color, alpha=0.8)
        ax.axvline(0, color="k", linewidth=1)
        ax.axvline(vals.mean(), color="k", linestyle="--", linewidth=1.2,
                   label=f"mean = {vals.mean():.1f}")
        ax.set_title(f"{name}\n(std = {vals.std():.1f}, {frac_neg:.0%} step downhill)")
        ax.set_xlabel("uphill component (projection onto mean direction)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("count")
    plt.tight_layout()
    plt.show()


def plot_aim_policy(xs, mu, sigma, target, title=None, ax=None):
    """Plot the AimEnv target function and the current Gaussian policy.

    Args:
        xs: 1-D array of states at which the policy was evaluated.
        mu: policy mean at each state.
        sigma: policy standard deviation (scalar or per-state array).
        target: the environment's optimal action f(x) at each state.
        title: optional axes title.
        ax: axes to draw on (defaults to current axes).
    """
    ax = ax or plt.gca()
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), np.shape(mu))
    ax.plot(xs, target, "k--", linewidth=1.8, label=r"optimal action $f(x)$")
    ax.plot(xs, mu, color="C0", linewidth=2.2, label=r"policy mean $\mu_\theta(x)$")
    ax.fill_between(xs, mu - sigma, mu + sigma, color="C0", alpha=0.25,
                    label=r"$\mu_\theta(x) \pm \sigma_\theta$")
    ax.set_xlabel("state $x$")
    ax.set_ylabel("action $a$")
    ax.set_ylim(-3.2, 3.2)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
