"""
Experiment 2 — external curvature validation and the reserve frontier.

    python -m scripts.run_frontier --master data/master_dataset.csv --out results/

The curvature check needs master_dataset.csv (from src.data_collection). The
frontier runs standalone.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from src import simulator as S

POLICIES = {
    "release-on-arrival": np.full(40, 1.0),
    "throughput":         np.clip(np.linspace(0.2, 1.0, 40), 0, 1),
    "conserve":           np.full(40, 0.5),
}


def curvature_validation(master_path: str) -> dict:
    m = pd.read_csv(master_path, parse_dates=["date"])
    bd = m[m["SOFR"].notna()].dropna(subset=["sofr99_iorb_bps", "WRESBAL_bn"])
    xe = np.log(bd["WRESBAL_bn"]); be = np.polyfit(xe, bd["sofr99_iorb_bps"], 2)
    sim = [(mm, np.mean([S.simulate(s, mm, np.full(40, 0.5), collect_obs=False)["vw_delay"]
                         for s in range(1, 9)]))
           for mm in [0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0]]
    sim = np.array(sim); xs, ys = np.log(sim[:, 0]), sim[:, 1]
    bs = np.polyfit(xs, ys, 2)
    r2s = 1 - ((ys - np.polyval(bs, xs)) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
    return {"empirical_curvature": float(be[0]),
            "simulated_curvature": float(bs[0]),
            "simulated_convex_r2": float(r2s),
            "sim_points": sim.tolist()}


def min_reserve(ap, seeds, target=0.10) -> float:
    lo, hi = 0.2, 3.0
    for _ in range(13):
        mid = (lo + hi) / 2
        u = np.mean([S.simulate(s, mid, ap, collect_obs=False)["unsettled"] for s in seeds])
        if u <= target:
            hi = mid
        else:
            lo = mid
    return hi


def frontier() -> dict:
    per = {k: [min_reserve(v, [s]) for s in range(1, 9)] for k, v in POLICIES.items()}
    w, p = stats.wilcoxon(np.array(per["throughput"]) - np.array(per["release-on-arrival"]))
    saving = 100 * (np.mean(per["throughput"]) - np.mean(per["release-on-arrival"])) / np.mean(per["throughput"])
    coll = {k: float(np.mean([S.simulate(s, float(np.mean(v)), POLICIES[k], collect_obs=False)["coll"]
                              for s in range(1, 9)])) for k, v in per.items()}
    return {"min_reserve": {k: [float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))]
                            for k, v in per.items()},
            "onarrival_vs_throughput_pct": float(saving),
            "wilcoxon_p": float(p),
            "collateral_at_operating_point": coll}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="data/master_dataset.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    result = {"frontier": frontier()}
    if Path(args.master).exists():
        result["curvature"] = curvature_validation(args.master)
        print("empirical curvature %+.1f | simulated %+.2f (convex R2 %.3f)" % (
            result["curvature"]["empirical_curvature"],
            result["curvature"]["simulated_curvature"],
            result["curvature"]["simulated_convex_r2"]))
    else:
        print("master_dataset.csv not found — skipping curvature validation")

    f = result["frontier"]
    print(f"release-on-arrival saves {f['onarrival_vs_throughput_pct']:.1f}% reserve "
          f"(Wilcoxon p={f['wilcoxon_p']:.4f})")
    json.dump(result, open(out / "frontier.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
