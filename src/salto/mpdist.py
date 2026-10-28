import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .data import znorm

THRESHOLD = 0.05


def _windows(X, l):
    return znorm(sliding_window_view(np.atleast_2d(np.asarray(X, dtype=np.float64)), l, axis=-1))


def mpdist_matrix(A, B, l, threshold=THRESHOLD, chunk=64):
    WA, WB = _windows(A, l), _windows(B, l)
    k = max(1, int(threshold * (WA.shape[1] + WB.shape[1])))
    WBt = WB.transpose(0, 2, 1)[None]
    out = np.empty((len(WA), len(WB)))
    for s in range(0, len(WA), chunk):
        corr = np.clip(WA[s:s + chunk, None] @ WBt / l, -1.0, 1.0)
        P = np.concatenate([corr.max(axis=3), corr.max(axis=2)], axis=2)
        c = -np.partition(-P, k - 1, axis=2)[..., k - 1]
        out[s:s + chunk] = np.sqrt(np.maximum(2 * l * (1 - c), 0.0))
    return out
