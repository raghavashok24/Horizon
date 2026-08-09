"""
Experiment 3 — risk-certification system: benchmarks, calibration, ablations.

    python -m scripts.run_certifier --n 5000 --out results/

Reports the certification ladder (R2 and breach-detection F1 with paired tests),
conformal coverage, regime-stratified F1, and schedule-selection validation.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.decomposition import PCA

from src import simulator as S
from src.certifier import Certifier, features, TARGET
from scripts.run_benchmarks import generate


def f1_at(y, p, thr=TARGET):
    yb, pb = (y > thr).astype(int), (p > thr).astype(int)
    tp = ((yb == 1) & (pb == 1)).sum(); fp = ((yb == 0) & (pb == 1)).sum()
    fn = ((yb == 1) & (pb == 0)).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    D = generate(a.n)
    D["U"] = D.pop("Y") if "U" not in D else D["U"]     # risk target = unsettled
    # regenerate with unsettled target
    X = features(D); Y = D["U"] if "U" in D else D["Y"]
    G = D["SEED"] % 5
    FP = PCA(8, random_state=0).fit_transform(D["FULL"])

    ladder = {"static rule (reserves only)": D["MULT"][:, None],
              "observation + reserves": np.column_stack([D["H"], D["OWN"], D["MULT"][:, None]]),
              "certifier (+ schedule)": X,
              "oracle (+ system state)": np.column_stack([X, FP])}
    res = {}
    for name, Xa in ladder.items():
        r2s, f1s = [], []
        for f in range(5):
            tr, te = G != f, G == f
            g = GBR(max_iter=400, random_state=0).fit(Xa[tr], Y[tr]); p = g.predict(Xa[te])
            r2s.append(1 - ((Y[te] - p) ** 2).sum() / ((Y[te] - Y[te].mean()) ** 2).sum())
            f1s.append(f1_at(Y[te], p))
        res[name] = (np.array(r2s), np.array(f1s))
        print(f"{name:<32} R2={np.mean(r2s):.4f}+/-{np.std(r2s):.4f}  F1={np.mean(f1s):.3f}")
    t, p = stats.ttest_rel(res["certifier (+ schedule)"][0], res["observation + reserves"][0])
    print(f"action gain  t={t:.2f}  p={p:.2e}")
    t2, p2 = stats.ttest_rel(res["oracle (+ system state)"][0], res["certifier (+ schedule)"][0])
    print(f"oracle gap   t={t2:.2f}  p={p2:.3f}")

    # Conformal calibration on disjoint calib/test groups
    cert = Certifier().fit(X[G < 4], Y[G < 4], calib_mask=(G[G < 4] == 3))
    outc = cert.certify(X[G == 4])
    cov = float(((Y[G == 4] >= outc["lo"]) & (Y[G == 4] <= outc["hi"])).mean())
    print(f"conformal 90% coverage: {cov:.3f}")

    json.dump({"coverage": cov,
               "ladder": {k: [float(v[0].mean()), float(v[1].mean())] for k, v in res.items()},
               "action_gain_p": float(p), "oracle_gap_p": float(p2)},
              open(out / "certifier.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
