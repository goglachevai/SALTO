import numpy as np
import pycatch22

N_FEATURES = 22


def c22_features(x):
    values = pycatch22.catch22_all(np.asarray(x, dtype=np.float64).tolist())["values"]
    return np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def c22_features_batch(X):
    return np.stack([c22_features(x) for x in X])


class C22Space:
    def __init__(self, fmin, fmax):
        self.fmin = np.asarray(fmin, dtype=np.float64)
        self.fmax = np.asarray(fmax, dtype=np.float64)
        rng = self.fmax - self.fmin
        self._const = rng <= 0
        self._rng = np.where(self._const, 1.0, rng)

    @classmethod
    def fit(cls, F):
        return cls(F.min(axis=0), F.max(axis=0))

    def transform(self, F):
        Fh = (np.asarray(F, dtype=np.float64) - self.fmin) / self._rng
        Fh[..., self._const] = 0.0
        return Fh

    @staticmethod
    def dist(a, B):
        return np.sqrt(((B - a) ** 2).sum(axis=-1))

    def state_dict(self):
        return {"fmin": self.fmin, "fmax": self.fmax}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["fmin"], d["fmax"])
