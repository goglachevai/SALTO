import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from salto import (METRICS, C22Space, Extractor, Salto, block_split, build_training_set, c22_features_batch,
                   load_pastila, load_series, rmse_loss, subsequences, znorm)
from paths import DEFAULT_MODEL, DEFAULT_RESULTS, DEFAULT_SERIES


def macro_f1(y_true, y_pred, K):
    f1 = []
    for k in range(K):
        tp = np.sum((y_pred == k) & (y_true == k))
        fp = np.sum((y_pred == k) & (y_true != k))
        fn = np.sum((y_pred != k) & (y_true == k))
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1))


def scores(y_true, y_pred, K):
    return {"accuracy": float(np.mean(y_true == y_pred)), "f1_macro": macro_f1(y_true, y_pred, K)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--series", default=str(DEFAULT_SERIES))
    p.add_argument("--results", default=str(DEFAULT_RESULTS))
    p.add_argument("--out", default=str(DEFAULT_MODEL))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--gamma", type=float, default=0.99, help="экспоненциальное затухание lr за эпоху")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--block", type=int, default=0, help="длина блока разбиения (по умолчанию 2*m)")
    p.add_argument("--full", action="store_true", help="обучать на всем фрагменте (без отложенных блоков)")
    p.add_argument("--balance", action=argparse.BooleanOptionalAction, default=True,
                   help="равновероятная выборка классов в эпохе")
    p.add_argument("--metric", choices=METRICS, default="c22",
                   help="расстояние для выбора ближайшего сниппета: c22 (C22dist) или mpdist (MPdist)")
    p.add_argument("--mp-window", type=int, default=0,
                   help="длина подпоследовательностей внутри MPdist (по умолчанию m/2, как в PSF)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    T = load_series(args.series)
    res = load_pastila(args.results)
    X, Y, L = build_training_set(T, res)
    m, K = res.m, res.K
    print(f"n={len(T)}, m_opt={m}, K={K}, |D|={len(X)}, доли классов={np.bincount(L, minlength=K) / len(L)}")

    space = C22Space.fit(c22_features_batch(subsequences(T, m))) if args.metric == "c22" else None

    if args.full:
        tr, va, te = np.arange(len(X)), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    else:
        tr, va, te = block_split(L, m, args.block or 2 * m, seed=args.seed)
    print(f"train={len(tr)} val={len(va)} test={len(te)}")

    dev = torch.device(args.device)
    Xt = torch.as_tensor(znorm(X), dtype=torch.float32, device=dev)
    Yt = torch.as_tensor(znorm(Y), dtype=torch.float32, device=dev)

    model = Extractor(m).to(dev)
    salto = Salto(model, res.snippets, space, device=args.device, metric=args.metric, mp_window=args.mp_window)
    print(f"параметров: {sum(p.numel() for p in model.parameters())}, метрика: {args.metric}"
          + (f" (окно {salto.mp_window})" if args.metric == "mpdist" else ""))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, args.gamma)
    monitor = va if len(va) else tr
    sample_p = 1.0 / np.bincount(L[tr], minlength=K)[L[tr]]
    sample_p /= sample_p.sum()

    best, best_state, wait = np.inf, None, 0
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.balance:
            perm = np.random.choice(tr, size=len(tr), p=sample_p)
        else:
            perm = tr[np.random.permutation(len(tr))]
        tot = 0.0
        for s in range(0, len(perm), args.batch):
            b = torch.as_tensor(perm[s:s + args.batch], device=dev)
            loss = rmse_loss(model(Xt[b]), Yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        sched.step()

        model.eval()
        with torch.no_grad():
            mon = torch.as_tensor(monitor, device=dev)
            C = model(Xt[mon])
            val = rmse_loss(C, Yt[mon]).item()
        pred = np.argmin(salto.distances(C.cpu().numpy().astype(np.float64)), axis=1)
        print(f"эпоха {epoch:3d}  train RMSE={tot / len(tr):.4f}  val RMSE={val:.4f}  "
              f"val acc={np.mean(pred == L[monitor]):.4f}  lr={sched.get_last_lr()[0]:.2e}  "
              f"{time.perf_counter() - t0:.0f}с", flush=True)
        if val < best - 1e-5:
            best, wait = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print("ранняя остановка")
                break

    model.load_state_dict(best_state)
    model.eval()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    salto.save(args.out)

    metrics = {"train_time_s": time.perf_counter() - t0, "best_val_rmse": best}
    for name, idx in (("train", tr), ("val", va), ("test", te), ("all", np.arange(len(X)))):
        if len(idx):
            metrics[name] = scores(L[idx], salto.classify_batch(X[idx]), K)
            print(name, metrics[name])
    with open(args.out.rsplit(".", 1)[0] + "_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
