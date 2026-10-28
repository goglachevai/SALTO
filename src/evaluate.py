import argparse
import time

import numpy as np
import torch

from paths import DEFAULT_MODEL, DEFAULT_RESULTS, DEFAULT_SERIES
from salto import METRICS, Salto, build_training_set, load_pastila, load_series
from train import scores


def ground_truth(n_windows, m, change_points):
    centers = np.arange(n_windows) + m // 2
    return np.searchsorted(np.asarray(change_points), centers, side="right")


def benchmark(salto, X, n, warmup=50):
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(X), n + warmup)
    times = []
    for j, i in enumerate(idx):
        t = time.perf_counter()
        salto.classify(X[i])
        if j >= warmup:
            times.append(time.perf_counter() - t)
    t = np.array(times) * 1e3
    return {"mean_ms": t.mean(), "p50_ms": np.percentile(t, 50), "p99_ms": np.percentile(t, 99), "max_ms": t.max()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--series", default=str(DEFAULT_SERIES))
    p.add_argument("--results", default=str(DEFAULT_RESULTS))
    p.add_argument("--change-points", type=int, nargs="*", default=[],
                   help="точки смены активности в ряде для сравнения с истинной разметкой")
    p.add_argument("--n", type=int, default=1000, help="число подпоследовательностей для замера времени")
    p.add_argument("--metric", choices=METRICS, default=None,
                   help="переопределить метрику выбора сниппета (по умолчанию — сохраненная в модели)")
    p.add_argument("--mp-window", type=int, default=0, help="длина подпоследовательностей внутри MPdist")
    p.add_argument("--devices", nargs="*", default=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
    args = p.parse_args()

    T = load_series(args.series)
    res = load_pastila(args.results)
    torch.set_num_threads(1)
    salto = Salto.load(args.model, device=args.devices[-1], metric=args.metric, mp_window=args.mp_window)
    print(f"метрика: {salto.metric}" + (f" (окно {salto.mp_window})" if salto.metric == "mpdist" else ""))
    X = build_training_set(T, res)[0]

    pred = salto.classify_batch(X)
    print("SALTO vs PaSTiLa:", scores(res.labels, pred, salto.K))
    if args.change_points:
        gt = ground_truth(len(X), salto.m, args.change_points)
        mapping = np.array([np.bincount(gt[res.labels == k], minlength=gt.max() + 1).argmax()
                            for k in range(salto.K)])
        print("PaSTiLa vs истинная:", scores(gt, mapping[res.labels], gt.max() + 1))
        print("SALTO   vs истинная:", scores(gt, mapping[pred], gt.max() + 1))

    for dev in args.devices:
        s = Salto.load(args.model, device=dev, metric=args.metric, mp_window=args.mp_window)
        r = benchmark(s, X, args.n)
        print(f"время вывода [{dev}]: " + ", ".join(f"{k}={v:.2f}" for k, v in r.items()))


if __name__ == "__main__":
    main()
