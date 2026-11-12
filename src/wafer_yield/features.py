"""Engineered features for wafer map defect pattern classification.

Each wafer map is turned into a fixed-length vector, so maps of any grid
size (WM-811K has 632 shapes) can go into one model.

Feature groups (85 features in total):
    fail_rate           share of tested dies that failed
    grid_r_c (64)       fail rate in each cell of an 8 x 8 grid laid over the
                        wafer, in normalized coordinates. Captures where on
                        the wafer the failures sit.
    ring_k (5)          fail rate in 5 concentric rings, center to edge.
                        Separates Center, Donut, and Edge-Ring.
    sector_k (8)        fail rate in 8 angular sectors. Separates a localized
                        Edge-Loc from a full Edge-Ring.
    edge_ratio          outer-ring fail rate / overall fail rate
    fail_r_mean, fail_r_std
                        mean and spread of failing dies' distance from center
    n_clusters, largest_cluster_frac, largest_cluster_elong
                        connected groups of failing dies. Elongation near 1
                        means a line, which is how Scratch looks.
    die_count           number of tested dies (proxy for product type)
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage

GRID = 8
RINGS = 5
SECTORS = 8
EIGHT_CONNECTED = np.ones((3, 3), dtype=int)


def feature_names() -> list[str]:
    names = ["fail_rate"]
    names += [f"grid_{r}_{c}" for r in range(GRID) for c in range(GRID)]
    names += [f"ring_{k}" for k in range(RINGS)]
    names += [f"sector_{k}" for k in range(SECTORS)]
    names += ["edge_ratio", "fail_r_mean", "fail_r_std",
              "n_clusters", "largest_cluster_frac", "largest_cluster_elong", "die_count"]
    return names


N_FEATURES = len(feature_names())


@lru_cache(maxsize=2048)
def _geometry(h: int, w: int):
    """Coordinates for an h x w grid, computed once per map shape."""
    yy, xx = np.mgrid[:h, :w].astype(float)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    dy, dx = yy - cy, xx - cx
    theta = np.arctan2(dy, dx)  # -pi..pi
    sector = np.minimum(((theta + np.pi) / (2 * np.pi) * SECTORS).astype(int), SECTORS - 1)
    gy = np.minimum((yy / h * GRID).astype(int), GRID - 1)
    gx = np.minimum((xx / w * GRID).astype(int), GRID - 1)
    grid = gy * GRID + gx
    return dy, dx, sector.ravel(), grid.ravel()


def _rate(fail_counts: np.ndarray, die_counts: np.ndarray) -> np.ndarray:
    out = np.zeros_like(fail_counts, dtype=float)
    np.divide(fail_counts, die_counts, out=out, where=die_counts > 0)
    return out


def wafer_features(m: np.ndarray) -> np.ndarray:
    """Feature vector for one wafer map (0 = no die, 1 = pass, 2 = fail)."""
    m = np.asarray(m)
    h, w = m.shape
    dy, dx, sector, grid = _geometry(h, w)
    die = (m > 0).ravel()
    fail = (m == 2).ravel()
    n_die = int(die.sum())
    f = np.zeros(N_FEATURES)
    if n_die == 0:
        return f

    # Radius normalized so the outermost die on this wafer sits at 1.
    r = np.sqrt(dy * dy + dx * dx).ravel()
    r_max = r[die].max() or 1.0
    r = r / r_max
    ring = np.minimum((r * RINGS).astype(int), RINGS - 1)

    fail_rate = fail.sum() / n_die
    i = 0
    f[i] = fail_rate; i += 1

    f[i:i + GRID * GRID] = _rate(np.bincount(grid[fail], minlength=GRID * GRID),
                                 np.bincount(grid[die], minlength=GRID * GRID))
    i += GRID * GRID

    ring_rate = _rate(np.bincount(ring[fail], minlength=RINGS), np.bincount(ring[die], minlength=RINGS))
    f[i:i + RINGS] = ring_rate; i += RINGS
    f[i:i + SECTORS] = _rate(np.bincount(sector[fail], minlength=SECTORS),
                             np.bincount(sector[die], minlength=SECTORS))
    i += SECTORS

    f[i] = ring_rate[-1] / fail_rate if fail_rate > 0 else 0.0; i += 1
    if fail.any():
        f[i] = r[fail].mean(); f[i + 1] = r[fail].std()
    i += 2

    # Connected clusters of failing dies.
    labels, n = ndimage.label(m == 2, structure=EIGHT_CONNECTED)
    f[i] = n
    if n > 0:
        sizes = np.bincount(labels.ravel())[1:]
        big = int(sizes.argmax()) + 1
        f[i + 1] = sizes.max() / max(fail.sum(), 1)
        ys, xs = np.nonzero(labels == big)
        if len(ys) >= 3:
            ev = np.linalg.eigvalsh(np.cov(np.vstack([ys, xs])))
            f[i + 2] = 1 - ev[0] / ev[1] if ev[1] > 0 else 0.0  # 1 = line, 0 = round blob
    i += 3

    f[i] = n_die
    return f


def build_matrix(maps, progress_every: int = 50_000) -> np.ndarray:
    """Stack features for many maps into an (n_wafers, N_FEATURES) float32 array."""
    X = np.zeros((len(maps), N_FEATURES), dtype=np.float32)
    for k, m in enumerate(maps):
        X[k] = wafer_features(m)
        if progress_every and (k + 1) % progress_every == 0:
            print(f"  features for {k + 1:,} / {len(maps):,} wafers", flush=True)
    return X