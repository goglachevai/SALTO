from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class PastilaResult:
    snippets: np.ndarray
    indices: np.ndarray
    labels: np.ndarray
    fracs: np.ndarray
    profiles: np.ndarray

    @property
    def m(self):
        return self.snippets.shape[1]

    @property
    def K(self):
        return self.snippets.shape[0]


def load_series(path):
    return np.loadtxt(path, dtype=np.float64).ravel()


def load_pastila(results_dir):
    d = Path(results_dir)
    snippets = np.atleast_2d(np.loadtxt(d / "snippets.txt", dtype=np.float64))
    return PastilaResult(
        snippets=snippets,
        indices=np.atleast_1d(np.loadtxt(d / "indicies.txt")).astype(np.int64),
        labels=np.loadtxt(d / "labels.txt").astype(np.int64).ravel(),
        fracs=np.atleast_1d(np.loadtxt(d / "fracs.txt", dtype=np.float64)),
        profiles=np.atleast_2d(np.loadtxt(d / "profiles.txt", dtype=np.float64)),
    )


def subsequences(T, m):
    return sliding_window_view(T, m)


def znorm(X, eps=1e-8):
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True)
    return (X - mu) / np.where(sd < eps, 1.0, sd)


def build_training_set(T, res):
    X = subsequences(T, res.m)
    if len(res.labels) not in (len(X), len(X) - 1):
        raise ValueError(f"длина разметки {len(res.labels)} != n - m + 1 = {len(X)}")
    X = X[:len(res.labels)]
    Y = res.snippets[res.labels]
    return X, Y, res.labels


def block_split(labels, m, block, test_fraction=0.2, val_fraction=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n_windows = len(labels)
    n_blocks = int(np.ceil(n_windows / block))
    n_test = max(1, round(test_fraction * n_blocks))
    classes = np.unique(labels)
    for _ in range(1000):
        is_test = np.zeros(n_blocks, dtype=bool)
        is_test[rng.permutation(n_blocks)[:n_test]] = True
        test_mask = np.repeat(is_test, block)[:n_windows]
        if np.array_equal(np.unique(labels[test_mask]), classes):
            break

    near_test = np.zeros(n_windows, dtype=bool)
    for lo, hi in _runs(np.flatnonzero(test_mask)):
        near_test[max(0, lo - m + 1):min(n_windows, hi + m)] = True
    rest = rng.permutation(np.flatnonzero(~near_test))
    n_val = round(val_fraction * len(rest))
    return np.sort(rest[n_val:]), np.sort(rest[:n_val]), np.flatnonzero(test_mask)


def _runs(idx):
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts, ends))
