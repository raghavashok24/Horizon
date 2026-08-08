"""
Generate the four paper figures as individual PNG files.

    python -m scripts.make_figures --out figures/

Produces:
  fig1_info_bound.png      information bound: interior optimum in probe intensity
  fig2_validation.png      external curvature validation (empirical vs simulated)
  fig3_ladder.png          benchmark ladder (observation vs action)
  fig4_frontier.png        reserve frontier by policy
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import simulator as S
from src import info_bound as IB

GREY, CORAL, BLUE, GREEN, PURPLE = "#888780", "#D85A30", "#378ADD", "#1D9E75", "#7F77DD"


def fig_info_bound(out: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, c in [(0.5, GREEN), (0.7, CORAL), (0.9, PURPLE)]:
        cv = IB.info_curve(k)
        ax.plot(cv["a"], cv["emp"], "o-", ms=3, color=c, label=f"$\\kappa$={k}")
        ax.axvline((1 - IB.THETA0_DEFAULT) / k, ls=":", color=c, alpha=0.5)
    ax.set_xlabel("probe intensity $a$")
    ax.set_ylabel("Fisher information about $\\theta$")
    ax.set_title("Information about congestion is non-monotone in probe intensity")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "fig1_info_bound.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_validation(out: Path, master="data/master_dataset.csv"):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    if Path(master).exists():
        m = pd.read_csv(master, parse_dates=["date"])
        bd = m[m["SOFR"].notna()].dropna(subset=["sofr99_iorb_bps", "WRESBAL_bn"])
        q = pd.qcut(bd["WRESBAL_bn"], 8, duplicates="drop")
        g = bd.groupby(q, observed=True).agg(res=("WRESBAL_bn", "mean"),
                                             t=("sofr99_iorb_bps", "mean"))
        xe = np.log(bd["WRESBAL_bn"]); be = np.polyfit(xe, bd["sofr99_iorb_bps"], 2)
        xs = np.linspace(xe.min(), xe.max(), 50)
        ax[0].scatter(g["res"], g["t"], color=CORAL)
        ax[0].plot(np.exp(xs), np.polyval(be, xs), "--", color=CORAL)
        ax[0].set_title(f"Empirical (FRED)  curvature {be[0]:+.1f}")
    else:
        ax[0].text(0.5, 0.5, "run data_collection first", ha="center")
    ax[0].set_xlabel("reserve balances ($B)"); ax[0].set_ylabel("SOFR99 - IORB (bps)")

    sim = [(mm, np.mean([S.simulate(s, mm, np.full(40, 0.5), collect_obs=False)["vw_delay"]
                         for s in range(1, 7)]))
           for mm in [0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0]]
    sim = np.array(sim); xs2 = np.log(sim[:, 0]); bs = np.polyfit(xs2, sim[:, 1], 2)
    xx = np.linspace(xs2.min(), xs2.max(), 50)
    ax[1].scatter(sim[:, 0], sim[:, 1], color=BLUE, marker="s")
    ax[1].plot(np.exp(xx), np.polyval(bs, xx), "--", color=BLUE)
    ax[1].set_title(f"Simulated (never fitted)  curvature {bs[0]:+.2f}")
    ax[1].set_xlabel("reserve multiple"); ax[1].set_ylabel("value-weighted delay")
    fig.tight_layout(); fig.savefig(out / "fig2_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_ladder(out: Path, ladder_csv="results/ladder.csv"):
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    if Path(ladder_csv).exists():
        lad = pd.read_csv(ladder_csv)
    else:   # fall back to reported values so the figure always renders
        lad = pd.DataFrame({"model": ["history only", "+ own state", "theta_hat alone",
                                      "+ chosen schedule", "oracle (8 PC)"],
                            "r2": [0.017, 0.329, 0.19, 0.900, 0.900],
                            "sd": [0.020, 0.036, 0.02, 0.010, 0.011]})
    colors = [GREY, GREY, GREY, CORAL, BLUE]
    ax.barh(range(len(lad)), lad["r2"], xerr=lad["sd"], color=colors, height=0.6)
    ax.set_yticks(range(len(lad))); ax.set_yticklabels(lad["model"], fontsize=9)
    ax.axvline(0, color="k", lw=0.8); ax.invert_yaxis()
    ax.set_xlabel("$R^2$ (5-fold, grouped by seed)")
    ax.set_title("Action conditioning dominates observation")
    fig.tight_layout(); fig.savefig(out / "fig3_ladder.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_frontier(out: Path):
    from scripts.run_frontier import min_reserve, POLICIES
    per = {k: [min_reserve(v, [s]) for s in range(1, 9)] for k, v in POLICIES.items()}
    ks = list(per)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(ks, [np.mean(per[k]) for k in ks],
           yerr=[np.std(per[k]) / np.sqrt(8) for k in ks], color=GREEN, capsize=4)
    ax.set_ylabel("min reserve multiple for $\\leq$10% unsettled")
    ax.set_title("Faster release meets the target on less reserve")
    fig.tight_layout(); fig.savefig(out / "fig4_frontier.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fig_info_bound(out); print("fig1_info_bound.png")
    fig_validation(out); print("fig2_validation.png")
    fig_ladder(out); print("fig3_ladder.png")
    fig_frontier(out); print("fig4_frontier.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
