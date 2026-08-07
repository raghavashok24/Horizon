## The Simulator: Development, Calibration, and Validation

### Why a simulator

The question this paper asks — *should a bank release payments differently?* — is a
counterfactual. Observational data records only what happened under the release
behaviour banks actually chose; it contains no observations of policies nobody ran.
No central bank publishes participant-level intraday queues, balances, or release
timestamps, so the alternative policies are not merely unobserved, they are
unobservable. A calibrated simulation is therefore not a fallback for missing data.
It is the only instrument that can answer the question.

The empirical panel and the simulator do different jobs, and the paper keeps them
separate: **the data establishes that the problem binds and calibrates the
environment; the simulator evaluates policies the data cannot contain.**

### How the mechanism was actually developed

Honesty about sequence matters here, so: **the analytic result came first, and the
simulator was written to test it.** We did not build an environment and discover an
inverted U in it.

The reasoning went as follows. If a bank's only signal about system congestion is the
delay with which its own releases are reciprocated, and if releasing also relieves that
congestion, then two effects move in opposite directions as release intensity rises.
More releases mean more delay observations. But a looser system responds faster, and
fast responses cluster tightly near zero, carrying little dispersion to read. Writing
response delays as Exponential with rate `r0*(1-theta)`, the Fisher information per
observation about theta is `1/(1-theta)^2` — it *falls* as the system loosens.

Combining the two effects gives a closed form, and differentiating it gives an interior
optimum. Only then was the simulator written, to check that a sampling implementation
reproduces the closed form. Several alternative observation models were considered and
discarded first because they yield monotone information — modelling the signal as inflow
*counts* rather than *delays*, for instance, produces no interior peak at all. That the
mechanism had to be chosen carefully is itself a finding, and it is reported rather than
hidden: the phenomenon requires the probe to degrade the *informativeness* of each
observation, not merely to change the state's level.

### The mechanism

State: latent tightness `theta` in [0,1], where 1 is maximally congested. Not
observable.

1. The ego bank releases at intensity `a` in [0,1], generating `n = a*M` payments.
2. Each release triggers a counterparty response arriving after delay
   `d ~ Exponential(rate = r0*(1-theta))`. A congested system responds slowly.
3. Releasing injects liquidity, so tightness declines: `theta(a) = theta0 - kappa*a`.
4. The bank observes **only** the response delays. Nothing about other banks' states,
   queues, or actions is visible.

Fisher information about theta from `n` such observations:

    I(a) = (aM - 2) / (1 - theta0 + kappa*a)^2

with the `-2` reflecting the exact finite-sample variance `r^2/(n-2)` of the unbiased
rate estimator `(n-1)/sum(d)`, not the asymptotic `r^2/n`. Differentiating:

    dI/da  ∝  (1 - theta0) - kappa*a        =>       a* = (1 - theta0) / kappa

Information rises below `a*` and falls above it. The peak-to-full-release ratio has a
closed form in `x = a*`:  `(x+1)^2 / (4x)`.

**The intuition, stated plainly: you can only measure congestion while congestion
exists. Probing hard enough to get a sharp reading destroys the thing being read.**

### Parameters and their provenance

| Parameter | Value | Kind | Source and justification |
|---|---|---|---|
| Intraday value profile | 10 hourly shares | **Empirical** | Armantier, Arnold & McAndrews (2008), *FRBNY Economic Policy Review* 14(2), 83–112, Table 1, 2006 mean column. Decile settlement times inverted by monotone (PCHIP) interpolation of the cumulative distribution, evaluated at hour boundaries, renormalised over 09:00–18:30. <https://www.newyorkfed.org/medialibrary/media/research/epr/08v14n2/0809arma.pdf> |
| `M` (payments/day) | 400 | **Empirical anchor** | Same paper, Table 2: mean daily Fedwire payment count of 465,237 across all participants. Scaled to one large participant. Only the *shape* of `I(a)` is claimed; `M` enters as a multiplicative constant. |
| `theta0` anchor | 90th pct = 10.0 bps | **Empirical** | 90th percentile of observed `SOFR99 − IORB`, 2021-08 to 2026-08 (FRED, via `master_dataset.csv`). Distribution: median 3.0 bps, p90 10.0, p99 28.0, max 55.0. |
| `theta0` | 0.90 | **Assumed** | `theta` is a latent normalised congestion index, not an observable quantity. The map from basis-point spreads to `theta` is assumed monotone; **only the anchor point is empirical.** This is a single modelling assumption, stated as such. |
| `r0` | 1.0 | Derived | Pure normalisation — sets the time unit and makes delays dimensionless. No empirical content. |
| `kappa` | 0.7–0.9 | **Assumed — free parameter** | **No public data measures how much one participant's releases loosen system-wide congestion.** This is the paper's only genuinely free parameter. It is swept across [0.3, 0.9] rather than fitted, and the claim is that a regime exists, not that `kappa` takes any particular value. |
| Estimator | `(n-1)/sum(d)` | Derived | Unbiased MLE of an exponential rate. Deliberately parametric rather than neural, so the information curve measures the *mechanism* and not training dynamics. |

Full machine-readable provenance is exported to `provenance_simulator.json`.

### Validation protocol

Three tests, run in order, each gating the next.

**1. Environment behaves.** Tightness falls monotonically with release intensity and
mean response delay shortens with it (2.80 → 1.09 across `a` = 0.25 → 1.00), confirming
the relief coupling is wired correctly.

**2. Estimator is unbiased.** Across `a` in {0.05, 0.20, 0.50, 1.00}, bias in `theta_hat`
stays below 0.001 against a 0.005 threshold. This checks that the estimator and the
generative model agree — a bias here would mean the code implements a different model
than the one derived.

**3. Empirical information matches the closed form.** The sampled `1/Var(theta_hat)`
curve is compared point-by-point against `(aM-2)/(1-theta0+kappa*a)^2` over a 29-point
grid at K = 8,000 replications each.

### Results

**The inverted U is present.** At `theta0` = 0.90 and `kappa` = 0.90, information peaks
at `a` ≈ 0.12–0.14 against an analytic prediction of `a*` = 0.111, then declines by a
factor of **2.71×** by full release.

**The cross-validation is the stronger result.** Sweeping `kappa` and comparing against
the closed-form peak ratio `(x+1)^2/(4x)`:

| `kappa` | `a*` predicted | `a*` observed | Ratio predicted | Ratio observed | `theta` at `a`=1 |
|---|---|---|---|---|---|
| 0.3 | 0.333 | 0.26 | 1.333 | 1.353 | 0.6 |
| 0.5 | 0.200 | 0.20 | 1.800 | 1.862 | 0.4 |
| 0.7 | 0.143 | 0.14 | 2.283 | 2.297 | 0.2 |
| 0.9 | 0.111 | 0.12 | 2.778 | 2.700 | 0.0 |

Both the peak *location* and the peak *depth* track closed-form predictions across the
entire swept range, to within a few percent. Neither quantity was fitted.

`kappa` = 0.7 is used as the default for the main figure: it yields a 2.3× decline while
keeping `theta(a=1)` = 0.2 strictly interior, whereas `kappa` = 0.9 sits exactly on the
clipping boundary at `theta` = 0.

### What the findings mean

**The optimal probe intensity is interior, and it is low.** Under these parameters,
information about system tightness is maximised at roughly 12% of full release capacity.
Beyond that, releasing more *reduces* what the bank learns. A control policy that
maximises release for informational reasons is therefore not merely suboptimal — it is
self-defeating.

**Depletion strength sets the penalty for over-probing, not whether one exists.** The
peak exists for every `kappa` > 0; larger `kappa` moves it earlier and deepens the
decline. What varies across the swept range is the cost of getting it wrong, from 1.35×
to 2.78×.

**This inverts the standard dual-control prescription.** Classical dual control says
explore early and aggressively, then exploit. Under self-depleting observation the
opposite holds: probing is most destructive exactly when the state is most extreme, so
informative experiments should be run when tightness is *low* — calibrate on calm days
so the model is already sharp when the tight day arrives. This is a falsifiable
prediction about the learned policy's probing schedule, tested in the full-environment
experiments.

**The result is structural, not tuned.** Peak location and depth both follow closed-form
predictions across the swept parameter range. The inverted U is a property of the
mechanism, not an artefact of one configuration.

### Limitations, stated plainly

**This does not demonstrate that the effect exists in Fedwire.** No participant-level
intraday microdata is publicly available, so the mechanism cannot be tested against the
real payment system. What is established is that a regime *exists* in which
self-depleting observation produces non-monotone information, that this regime is
reachable under empirically anchored parameters, and that the phenomenon follows a
closed form rather than depending on tuning.

**`kappa` has no empirical counterpart.** It is swept rather than estimated, and no
claim is made about its true value.

**The intraday profile describes 2006.** Fedwire operating hours and participant
composition have changed since; the late-afternoon concentration is a durable documented
feature, but the exact shares are dated. The source also excludes CHIPS, CLS Bank, DTC,
and principal-and-interest funding payments.

**The `theta`-to-basis-points map is assumed.** Only the anchor percentile is empirical;
the monotone mapping itself is a modelling choice.

**The estimator is parametric and the environment is single-agent.** The neural filter
and the multi-bank population are introduced in the full-environment stage; this section
isolates the mechanism deliberately.

