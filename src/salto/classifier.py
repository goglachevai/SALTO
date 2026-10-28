from collections import deque

import numpy as np
import torch

from .c22 import C22Space, c22_features_batch
from .data import znorm
from .model import Extractor
from .mpdist import mpdist_matrix

METRICS = ("c22", "mpdist")


class Salto:
    def __init__(self, model, snippets, space=None, device="cpu", metric="c22", mp_window=None):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.snippets = np.asarray(snippets, dtype=np.float64)
        self.m = self.snippets.shape[1]
        self.space = space
        self.mp_window = mp_window or self.m // 2
        self.set_metric(metric)

    def set_metric(self, metric):
        if metric not in METRICS:
            raise ValueError(f"неизвестная метрика {metric!r}, доступны: {', '.join(METRICS)}")
        if metric == "c22":
            if self.space is None:
                raise ValueError("для C22dist нужны параметры нормализации признаков (модель обучена без них)")
            self.snippet_feats = self.space.transform(c22_features_batch(self.snippets))
        elif not 3 <= self.mp_window <= self.m:
            raise ValueError(f"окно MPdist {self.mp_window} должно быть в диапазоне [3, m={self.m}]")
        self.metric = metric

    @property
    def K(self):
        return len(self.snippets)

    @torch.inference_mode()
    def extract(self, X):
        x = torch.as_tensor(znorm(X), dtype=torch.float32, device=self.device)
        return self.model(x).cpu().numpy().astype(np.float64)

    def distances(self, C):
        if self.metric == "mpdist":
            return mpdist_matrix(C, self.snippets, self.mp_window)
        F = self.space.transform(c22_features_batch(C))
        return C22Space.dist(F[:, None, :], self.snippet_feats[None])

    def nearest(self, c_recov):
        d = self.distances(np.asarray(c_recov)[None])[0]
        return int(np.argmin(d)), d

    def classify(self, x):
        return self.nearest(self.extract(np.asarray(x)[None])[0])[0]

    def classify_batch(self, X, batch_size=512):
        labels = np.empty(len(X), dtype=np.int64)
        for s in range(0, len(X), batch_size):
            labels[s:s + batch_size] = np.argmin(self.distances(self.extract(X[s:s + batch_size])), axis=1)
        return labels

    def stream(self):
        return SaltoStream(self)

    def save(self, path):
        torch.save({
            "m": self.m,
            "model": self.model.state_dict(),
            "snippets": self.snippets,
            "space": None if self.space is None else self.space.state_dict(),
            "metric": self.metric,
            "mp_window": self.mp_window,
        }, path)

    @classmethod
    def load(cls, path, device="cpu", metric=None, mp_window=None):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = Extractor(ck["m"])
        model.load_state_dict(ck["model"])
        space = C22Space.from_state_dict(ck["space"]) if ck.get("space") is not None else None
        return cls(model, ck["snippets"], space, device,
                   metric=metric or ck.get("metric", "c22"),
                   mp_window=mp_window or ck.get("mp_window"))


class SaltoStream:
    def __init__(self, salto):
        self.salto = salto
        self.buf = deque(maxlen=salto.m)

    def push(self, value):
        self.buf.append(float(value))
        if len(self.buf) < self.salto.m:
            return None
        return self.salto.classify(np.fromiter(self.buf, dtype=np.float64, count=self.salto.m))
