# SPDX-FileCopyrightText: : 2024 Lukas Franken (Heavy ChatGPT support)
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

def _nearest_low_distance(mask: np.ndarray) -> np.ndarray:
    n = len(mask)
    dist = np.full(n, np.inf, dtype=float)

    last = -np.inf
    for i in range(n):
        if mask[i]:
            last = i
            dist[i] = 0.0
        else:
            dist[i] = min(dist[i], i - last)

    last = np.inf
    for i in range(n - 1, -1, -1):
        if mask[i]:
            last = i
            dist[i] = 0.0
        else:
            dist[i] = min(dist[i], last - i, dist[i])

    if not np.isfinite(dist).any():
        dist[:] = np.arange(n)[::-1]
    else:
        dist[~np.isfinite(dist)] = dist[np.isfinite(dist)].max()
    return dist


def transform_prices_expand_low_windows(
    s: pd.Series,
    target_mean: float,
    q_low: float = 0.30,
    floor_quantile: float = 0.05,
    half_life_steps: float = 3.0,
    clip_min: float = 0.0,
):
    """
    Lower the mean by expanding low-price windows via distance-weighted pull
    toward a soft floor. If target_mean >= current mean, this function returns
    the original series unchanged (use the wrapper below to lift the mean).
    """
    if len(s) < 2:
        return s.copy(), {
            'current_mean': float(np.mean(s)),
            'target_mean': float(target_mean),
            'alpha': 0.0,
            'achieved_mean': float(np.mean(s)),
        }

    x = s.astype(float).to_numpy()
    current_mu = float(np.mean(x))

    if target_mean >= current_mu:
        return s.copy(), {
            'current_mean': current_mu,
            'target_mean': float(target_mean),
            'alpha': 0.0,
            'achieved_mean': current_mu,
        }

    thresh_low = np.quantile(x, q_low)
    low_mask = x <= thresh_low
    d = _nearest_low_distance(low_mask)

    half_life_steps = max(half_life_steps, 1e-6)
    w = 0.5 ** (d / half_life_steps)

    floor = np.quantile(x, floor_quantile)
    margin = np.maximum(0.0, x - floor)

    denom = float(np.mean(w * margin))
    if denom <= 0.0:
        return s.copy(), {
            'current_mean': current_mu,
            'target_mean': float(target_mean),
            'alpha': 0.0,
            'achieved_mean': current_mu,
        }

    alpha = (current_mu - target_mean) / denom
    alpha = max(0.0, min(1.0, alpha))

    y = x - alpha * w * margin
    if clip_min is not None:
        y = np.maximum(clip_min, y)

    y_series = pd.Series(y, index=s.index)
    achieved = float(y_series.mean())
    return y_series, {
        'current_mean': current_mu,
        'target_mean': float(target_mean),
        'alpha': float(alpha),
        'achieved_mean': achieved,
        'mode': 'lower_mean_expand_low_windows',
    }


def transform_prices_to_target_mean(
    s: pd.Series,
    target_mean: float,
    *,
    # params for the "lower mean" path:
    q_low: float = 0.30,
    floor_quantile: float = 0.05,
    half_life_steps: float = 3.0,
    clip_min: float = 0.0,
    # params for the "raise mean" path:
    clip_after_scale_min: float | None = 0.0,
):
    """
    Transform a price series so its mean hits `target_mean`.
    - If target_mean < current_mean: use the distance-weighted pull that expands
      low-price windows (preserves original logic).
    - If target_mean >= current_mean: simply scale all prices by a factor so the
      mean matches the target (simple, monotone lift).

    Returns
    -------
    y : pd.Series
    info : dict
        Includes 'mode' ('raise_mean_scale_all' or 'lower_mean_expand_low_windows'),
        'factor' (for raise), or 'alpha' (for lower), and achieved mean.
    """
    x = s.astype(float).to_numpy()
    current_mu = float(np.mean(x))

    if target_mean < current_mu:
        return transform_prices_expand_low_windows(
            s,
            target_mean=target_mean,
            q_low=q_low,
            floor_quantile=floor_quantile,
            half_life_steps=half_life_steps,
            clip_min=clip_min,
        )

    # --- Raise mean: simple multiplicative scaling ---
    # Handle degenerate (near-zero) mean safely
    eps = 1e-12
    denom = current_mu if abs(current_mu) > eps else eps
    factor = float(target_mean / denom)

    y = x * factor
    if clip_after_scale_min is not None:
        y = np.maximum(clip_after_scale_min, y)

    y_series = pd.Series(y, index=s.index)
    achieved = float(y_series.mean())
    return y_series, {
        'current_mean': current_mu,
        'target_mean': float(target_mean),
        'factor': factor,
        'achieved_mean': achieved,
        'mode': 'raise_mean_scale_all',
    }


# -------------- Example --------------
if __name__ == "__main__":
    rng = pd.date_range("2025-07-01", periods=24*7, freq="H")
    base = 60 + 10*np.sin(np.linspace(0, 8*np.pi, len(rng)))
    noise = np.random.default_rng(0).normal(0, 6, len(rng))
    s = pd.Series(base + noise, index=rng)
    s[s.index.hour.between(10, 15)] -= 20

    # Lower mean (expands low windows)
    down_target = s.mean() - 8
    y_down, info_down = transform_prices_to_target_mean(s, down_target)
    print(info_down)

    # Raise mean (simple scale)
    up_target = s.mean() + 8
    y_up, info_up = transform_prices_to_target_mean(s, up_target)
    print(info_up)
