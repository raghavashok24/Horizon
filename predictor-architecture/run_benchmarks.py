"""
Experiment 1 — benchmark ladder and ablations.

Generates rollouts, then evaluates the observation-vs-action dissociation with
5-fold CV grouped by seed and paired t-tests. Also runs the counterfactual and
regime ablations and the arrival-profile sensitivity sweep.

    python -m scripts.run_benchmarks --n 5000 --out results/

Writes results/ladder.csv, results/ablations.json.
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


def generate(n: int, seed0: int = 0) -> dict:
    rng = np.random.default_rng(seed0)
    rows = []
    for s in range(n):
        m = float(rng.uniform(0.2, 2.0))
        b = float(rng.uniform(0.05, 1.0)); sl = float(rng.uniform(-0.5, 0.5))
        ap = np.clip(b + sl * np.linspace(0, 1, 40), 0.02, 1.0)
        st = int(rng.integers(S.T0 + 2, 32)) if rng.random() < 0.6 else None
        sb = int(rng.integers(0, S.N_BANKS)) if (st is not None and rng.random() < 0.5) else None
        sg = float(rng.uniform(0.7, 1.6)) if st is not None else 1.0
        o = S.simulate(s, m, ap, sh_t=st, sh_b=sb, surge=sg)
        rows.append(dict(H=o["H"], OWN=o["OWN"], FULL=o["FULL"], A=ap,
                         TH=S.theta_hat(o["WAITS"]), Y=o["vw_delay"],
                         MULT=m, SEED=s))
    D = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    ok = np.isfinite(D["TH"])
    return {k: v[ok] for k, v in D.items()}


def cv_r2(D, cols, k=5):
    X, Y, G = np.column_stack(cols), D["Y"], D["SEED"] % k
    o = []
    for f in range(k):
        tr, te = G != f, G == f
        g = GBR(max_iter=400, random_state=0).fit(X[tr], Y[tr])
        p = g.predict(X[te])
        o.append(1 - ((Y[te] - p) ** 2).sum() / ((Y[te] - Y[te].mean()) ** 2).sum())
    return np.array(o)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    D = generate(args.n)
    A4 = np.c_[D["A"].mean(1), D["A"][:, 0], D["A"][:, -1], D["A"].std(1)]
    FP = PCA(8, random_state=0).fit_transform(D["FULL"])

    ladder = {
        "history only":        [D["H"]],
        "+ own state":         [D["H"], D["OWN"]],
        "theta_hat alone":     [D["TH"][:, None]],
        "+ chosen schedule":   [D["H"], D["OWN"], A4],
        "oracle (8 PC)":       [D["H"], D["OWN"], A4, FP],
    }
    res = {name: cv_r2(D, cols) for name, cols in ladder.items()}
    lad = pd.DataFrame([(n, v.mean(), v.std()) for n, v in res.items()],
                       columns=["model", "r2", "sd"])
    print(lad.round(4).to_string(index=False))
    lad.to_csv(out / "ladder.csv", index=False)

    def paired(a, b):
        t, p = stats.ttest_rel(res[b], res[a])
        return {"delta": float(res[b].mean() - res[a].mean()), "t": float(t), "p": float(p)}

    ablations = {
        "corr_theta_outcome": float(np.corrcoef(D["TH"], D["Y"])[0, 1]),
        "action_gain": paired("+ own state", "+ chosen schedule"),
        "oracle_gap": paired("+ chosen schedule", "oracle (8 PC)"),
    }

    # Counterfactual bands
    mean_a = D["A"].mean(1); X = np.column_stack([D["H"], D["OWN"], A4]); cf = []
    for lo, hi, lab in [(.05, .2, "low"), (.2, .4, "mid-low"), (.4, .6, "mid"),
                        (.6, .8, "mid-high"), (.8, 1., "high")]:
        te = (mean_a >= lo) & (mean_a < hi); tr = ~te
        g = GBR(max_iter=400, random_state=0).fit(X[tr], D["Y"][tr])
        p, y = g.predict(X[te]), D["Y"][te]
        cf.append({"band": lab, "n": int(te.sum()),
                   "r2": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())})
    ablations["counterfactual_bands"] = cf

    # Regime stratification
    qs = pd.Series(pd.qcut(D["MULT"], 4, labels=["Q1", "Q2", "Q3", "Q4"]))
    Xn = np.column_stack([D["H"], D["OWN"]]); reg = []
    for lab in qs.cat.categories:
        m = (qs == lab).to_numpy(); G = D["SEED"][m] % 5; Ys = D["Y"][m]

        def sub(Xa):
            o = []
            for k in range(5):
                tr, te = G != k, G == k
                gg = GBR(max_iter=300, random_state=0).fit(Xa[m][tr], Ys[tr])
                pp = gg.predict(Xa[m][te])
                o.append(1 - ((Ys[te] - pp) ** 2).sum() / ((Ys[te] - Ys[te].mean()) ** 2).sum())
            return float(np.mean(o))
        reg.append({"regime": lab, "obs": sub(Xn), "act": sub(X)})
    ablations["regime"] = reg

    json.dump(ablations, open(out / "ablations.json", "w"), indent=2)
    print("action gain:", ablations["action_gain"])
    print("oracle gap :", ablations["oracle_gap"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
