# Horizon: An Action-Conditional System for Certifying Intraday Settlement Risk

Horizon is a research system for measuring and certifying **intraday liquidity risk in real-time gross settlement (RTGS) payment systems** (Fedwire-style). It combines an empirical macro-funding dataset built from public Fed and FDIC data, an analytic information bound on what a bank can learn about system congestion, a calibrated multi-bank settlement simulator, and a machine-learned **risk certifier** that predicts end-of-day unsettled-value risk with statistically valid (conformal) uncertainty intervals.

The central idea is that settlement risk is **action-conditional**: a bank's end-of-day risk depends not just on what it observes mid-morning, but on the payment-release schedule it *plans to execute*. Conditioning on the planned schedule is what makes risk certifiable.

## Key results

- **Certification requires the planned schedule.** A gradient-boosting certifier given mid-morning observables, reserves, *and the planned release schedule* reaches **R² ≈ 0.90** on end-of-day unsettled-value risk — matching a dimension-matched oracle with full system state — versus **R² ≈ 0.33** for the same model without the schedule.
- **The information bound is real and non-monotone.** Fisher information about latent congestion is an inverted-U in a bank's release intensity, peaking at an interior optimum `a* = (1 − θ₀)/κ` (~12% of capacity in the calibrated setting). Probing hard enough to get a sharp reading relieves the congestion being measured — you can only measure congestion while congestion exists. The closed form is verified by Monte-Carlo simulation across a κ-sensitivity sweep.
- **Conformal calibration restores validity.** Raw quantile-regression intervals cover only ~72% at a nominal 90%; conformalized quantile regression (CQR) restores empirical coverage to ~90%.
- **Static reserve rules fail exactly where reserves are ample.** Breach-detection F1 for a static reserves-only rule falls from 1.00 to 0.70 moving from the scarcest to the most abundant reserve regime; the certifier stays at 0.91–1.00 across all regimes.
- **Schedule selection recovers release-on-arrival.** Choosing the lowest-certified-risk schedule concentrates on full-intensity release (84% in the top band), consistent with the analytic liquidity-recycling prediction, and meets a 10% unsettled-value target on materially less reserve than conservative policies.

## High Level of the Project

```
FRED + FDIC public data ──► empirical funding-tightness panel (2021–2026)
        │                        │
        │ calibrates             │ validates curvature (never fitted)
        ▼                        ▼
analytic information bound ◄──► 50-bank RTGS simulator ──► rollouts
   I(a) = (aM−2)/(1−θ₀+κa)²          │
                                     ▼
                     action-conditional certifier (HistGradientBoosting + CQR)
                                     │
                                     ▼
                 certified risk, valid 90% intervals, schedule selection
```

The data establishes that the problem binds and calibrates the environment; the simulator evaluates policies the data cannot contain. The analytic result came first, and the simulator was written to test it.

## Repository structure

| Directory | Contents |
|---|---|
| [`data-collection/`](data-collection/) | End-to-end scripted retrieval of the empirical dataset from two public APIs, the resulting master panel and stress-day table, and a full data-methodology README. |
| [`estimator-model/`](estimator-model/) | The self-depleting-observation information bound: analytic derivation, Monte-Carlo verification, parameter-provenance registry, and κ-sensitivity sweep. |
| [`predictor-tool/`](predictor-tool/) | The core library: calibrated N-bank RTGS simulator, latent-state estimator, and the action-conditional certifier with conformalized quantile regression. |
| [`predictor-architecture/`](predictor-architecture/) | Experiment scripts: benchmark ladder and ablations, external curvature validation, the reserve frontier, and figure generation. |
| [`finalized-figures/`](finalized-figures/) | The six finalized figures used in the paper. |
| [`submitted-materials/`](submitted-materials/) | Submitted paper materials (withheld until after review decision). |

## Data

All data is **public and free**, and the dataset reproduces end to end from two API calls. Sample window: **2021-08-01 to 2026-08-04** (1,830 calendar days / 1,249 business days; start set by the inception of the IORB series).

- **FRED** (St. Louis Fed API, free key required): 20 daily series covering reserve quantities (`WRESBAL`, `WALCL`, `WTREGEN`, …), open-market operations (`RPONTSYD`, `RRPONTSYD`), the policy corridor (`IORB`, `DFEDTARU`, `DFEDTARL`), the secured-rate distribution (`SOFR` and its 1st/25th/75th/99th percentiles, volumes), and unsecured rates (`EFFR`, `OBFR`).
- **FDIC BankFind Suite** (no key): quarterly Call Report items for all insured institutions, aggregated to size-distribution and cash-ratio statistics that parameterize the simulated bank population.
- **Armantier, Arnold & McAndrews (2008)**, *FRBNY Economic Policy Review* 14(2): published Fedwire intraday timing statistics used to calibrate the arrival process (no payment-level intraday data exists publicly).

The primary tightness measure throughout is **`SOFR99 − IORB`** — funding stress is a tail phenomenon, so the tail (the marginal borrower) is where it is observable. The pipeline includes documented validation checks (e.g., the $29.4B Standing Repo Facility draw on 2025-10-31 is reproduced exactly) and a documented units-bug fix with re-verified sanity checks. See [`data-collection/README.md`](data-collection/README.md) for full methodology, including sources that were evaluated and rejected.

Key artifacts: [`master_dataset (1).csv`](data-collection/) (1,830 × 36 daily panel) and [`stress_days.csv`](data-collection/stress_days.csv) (top-25 stress days).

## The information bound (`estimator-model/`)

A bank releasing payments at intensity `a` observes `n = aM` settlement delays drawn from an exponential with rate `r₀(1 − θ)`, but releasing also relieves congestion: `θ(a) = θ₀ − κa`. Fisher information about θ is then

```
I(a) = (aM − 2) / (1 − θ₀ + κa)²
```

which is an inverted-U with interior maximum `a* = (1 − θ₀)/κ`. Simulation confirms the peak at `a ≈ 0.12–0.14` against the analytic `0.111`, with information declining **2.7×** at full release. Every parameter carries recorded provenance (empirical / derived / assumed) in [`provenance_simulator.json`](estimator-model/provenance_simulator.json); the congestion-relief coefficient **κ is the paper's only genuinely free parameter and is swept (κ ∈ [0.3, 0.9]), not fitted**. The practical reading inverts the standard dual-control prescription: the best time to calibrate is calm days, and optimal probe intensity is interior and low.

## The simulator and certifier (`predictor-tool/`)

- **`simulator.py`** — a calibrated 50-bank RTGS environment: lognormal bank sizes fit to FDIC data, Fedwire intraday value profile, Fed PSR-style net-debit cap (35% of opening balance), Treasury collateral haircut, and full-day rollouts with optional post-decision shocks (demand surges, a stricken bank halting payments). Outputs realized waits, censoring-corrected value-weighted delay, unsettled-value fraction, and overdraft/collateral integrals.
- **`certifier.py`** — the action-conditional certifier: sklearn `HistGradientBoostingRegressor` point and quantile models over history, own-state, reserve, and schedule features, wrapped in **conformalized quantile regression** (Romano, Patterson & Candès, 2019) with a disjoint calibration split. `certify()` returns predicted risk, a valid 90% interval, and breach flags against a 10% unsettled-value target; `select_schedule()` picks the lowest-certified-risk release path among candidates.
- **`run_certifier.py`** — the certification-ladder experiment: static reserves-only → observation + reserves → certifier (+ schedule) → oracle (+ full system state), with 5-fold seed-grouped cross-validation, paired t-tests, and held-out conformal coverage.

Evaluation is deliberately conservative: seed-grouped splits (no rollout leakage), a disjoint conformal calibration set, and a dimension-matched (8-PC) oracle.

## Experiments (`predictor-architecture/`)

- **`run_benchmarks.py`** — 5,000 rollouts across random reserve multiples U(0.2, 2.0), random release schedules, and 60% shocked episodes; benchmark ladder, action-gain and oracle-gap ablations, counterfactual generalization across held-out schedule-intensity bands, and regime stratification by reserve quartile.
- **`run_frontier.py`** — external validation and the reserve frontier: the empirical convexity of SOFR99−IORB in log reserves is compared to the simulator's convexity of delay in log reserve-multiple (**never fitted to each other**), then bisection finds the minimum reserve multiple meeting the 10% target under release-on-arrival, ramp, and conserve policies.
- **`make_figures.py`** — regenerates the figures.

## Figures

The six finalized paper figures live in [`finalized-figures/`](finalized-figures/):

1. **Validation** — empirical (FRED) vs simulated curvature, independently convex.
2. **Certification ladder** — R² ≈ 0.39 / 0.33 / **0.90** / 0.90 across the four rungs; certification requires the planned schedule.
3. **Regime F1** — static rules fail exactly where reserves are ample; the certifier does not.
4. **Calibration & selection** — CQR restores 90% coverage; selection recovers release-on-arrival.
5. **Counterfactual generalization** — valid across most held-out schedule bands, with an honestly reported failure region at the lowest band.
6. **Sensitivity** — the action gain (ΔR² ≈ 0.42–0.48) is robust to the arrival-profile vintage and the daylight-credit cap.

## Reproducing

**Dependencies** (no pinned manifest yet): Python 3.10+, `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `requests`.

**Data**: open `data-collection/data_collection_script.ipynb` (written for Google Colab), insert a free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html), and run all cells to regenerate `master_dataset.csv`, `stress_days.csv`, and the tightness figure from scratch.

**Note on layout**: the experiment scripts were developed against a packaged layout and import as `from src import simulator` / `from scripts.run_benchmarks import generate` with data under `data/`. To run them from this repository, assemble that layout first:

```
project/
├── src/            # simulator.py, certifier.py from predictor-tool/
├── scripts/        # run_benchmarks.py, run_frontier.py, run_certifier.py, make_figures.py
└── data/           # master_dataset.csv (rename from "master_dataset (1).csv")
```

then run, e.g.:

```bash
python -m scripts.run_benchmarks --n 5000 --out results/
python -m scripts.run_frontier --master data/master_dataset.csv --out results/
python -m scripts.run_certifier --n 5000 --out results/
```

`estimator-model/estimator-script` is a Colab notebook export and is best run cell-by-cell in a notebook environment.

## Limitations

Stated plainly, as in the paper: the information-bound effect is demonstrated in a calibrated simulator, not in real Fedwire data (no payment-level intraday data is public); κ has no direct empirical counterpart and is swept rather than fitted; the intraday value profile is calibrated from 2006 Fedwire statistics; the θ-to-basis-points mapping is assumed; and the parametric estimation stage is single-agent. The empirical panel is system-level, not bank-level.

## Status

The accompanying paper is under review; submitted materials will be added to [`submitted-materials/`](submitted-materials/) after the decision.

## Author

**Shriraghav Ashok** ([@raghavashok24](https://github.com/raghavashok24))
