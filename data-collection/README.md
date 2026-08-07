
## Data Collection

All data used in this study is public and free. No proprietary, licensed, or
paid data sources are used. Retrieval is fully scripted; the cells above
reproduce the dataset end to end from two API calls.

### Sample window

**2021-08-01 to 2026-08-04** — 1,830 calendar days, of which 1,249 are business
days. The start date is set by data availability rather than choice: the
interest on reserve balances series (`IORB`) begins 2021-07-29, and IORB is the
reference rate against which every spread in this analysis is measured. Before
that date the equivalent construction would require splicing the legacy IOER
series, introducing a discontinuity we prefer to avoid.

### Source 1 — Federal Reserve Economic Data (FRED)

Federal Reserve Bank of St. Louis.
Endpoint: `https://api.stlouisfed.org/fred/series/observations`
Free API key: https://fredaccount.stlouisfed.org/apikeys

Twenty-two series were requested; **20 returned data**. `BGCR` (broad general
collateral rate) and `TGCR` (tri-party general collateral rate) failed and are
excluded. Both were intended as robustness checks on the secured-rate results
and neither is load-bearing, so no substitution was made.

| Group | Series retrieved | Role |
|---|---|---|
| Reserve quantity | `WRESBAL`, `WALCL`, `WTREGEN`, `WLCFLPCL` | The scarce resource. `WRESBAL` is the aggregate stock of reserve balances — the quantity the proposed policy economises on. Reported by FRED in **millions** of USD. |
| Open market operations | `RPONTSYD`, `RRPONTSYD` | Daily repo and reverse repo volumes. `RPONTSYD` identifies the dates on which the Standing Repo Facility was drawn, which anchor the stress episodes examined. |
| Policy corridor | `IORB`, `DFEDTARU`, `DFEDTARL` | IORB is the opportunity cost of holding reserves and the baseline from which funding tightness is measured. |
| Secured overnight rates | `SOFR`, `SOFR1`, `SOFR25`, `SOFR75`, `SOFR99`, `SOFRVOL` | The SOFR percentile distribution is the observable proxy for intraday funding pressure. |
| Unsecured rates | `EFFR`, `EFFR1`, `EFFR99`, `EFFRVOL`, `OBFR` | Robustness: confirms stress episodes identified in secured markets also appear in unsecured funding. |

**Why the percentile distribution and not the headline rate.** The volume-weighted
median SOFR reflects the typical borrower and moves little under intraday
pressure. The 99th percentile reflects the *marginal* borrower — the institution
that could not fund elsewhere and paid up. Funding stress is a tail phenomenon,
so the tail is where it is observable. `SOFR99 − IORB` is accordingly the primary
tightness measure throughout this paper.

**Handling of FRED conventions.** Missing observations are encoded as `"."` and
are converted to NaN rather than parsed as strings. Weekly series (`WRESBAL`,
`WALCL`, `WTREGEN`, `WLCFLPCL`, all Wednesday-stamped) are forward-filled onto
the daily index; rate series are business-day only and are left with gaps, with
an `is_business_day` flag retained instead of dropping rows.

### Source 2 — FDIC BankFind Suite

Federal Deposit Insurance Corporation.
Endpoint: `https://banks.data.fdic.gov/api/financials`
Documentation: https://banks.data.fdic.gov/docs/ — no API key required.

Institution-level quarterly Call Report items for the **eight most recent
quarter-end report dates** in the sample window, paginated at 5,000 records per
request. Fields retrieved: `CERT`, `REPDTE`, `NAME`, `ASSET`, `DEP`, `CHBAL`,
`CHBALI`.

Three quantities are aggregated per quarter and forward-filled onto the daily
index:

- `fdic_n_institutions` — count of reporting institutions
- `fdic_top10_asset_share` — assets of the ten largest as a share of the total
- `fdic_median_cash_ratio` — median of cash and balances due (`CHBAL`) divided by
  total assets, trimmed to (0, 1)

**Role.** These do not enter the empirical analysis directly. They parameterise
the simulated bank population, converting an arbitrary choice of N identical
agents into a population whose concentration and liquidity buffers match the
observed US banking system. The full institution-level panel is retained
separately as `fdic_institutions.csv` for any further distributional
calibration.

### Constructed variables

Merged on `date` by outer join, sorted, deduplicated.

**Spreads** (in basis points, computed as the rate difference × 100):

- `sofr99_iorb_bps` = SOFR99 − IORB — **primary tightness proxy**
- `sofr_iorb_bps` = SOFR − IORB — median-borrower comparison
- `sofr_tail_dispersion_bps` = SOFR99 − SOFR — how far the tail sits above the median
- `sofr_iqr_full_bps` = SOFR99 − SOFR1 — full observed range of the SOFR distribution
- `effr_iorb_bps` = EFFR − IORB — unsecured-market analogue

*Note: `sofr_iqr_full_bps` is a 1st-to-99th-percentile range, not an
interquartile range; the name is retained for continuity with the code.*

**Calendar flags:** `is_month_end`, `is_quarter_end`, `is_year_end`, `weekday`,
`is_business_day`.

**Unit normalisation:** `WRESBAL_bn = WRESBAL / 1000`. FRED reports `WRESBAL` in
millions, while `RPONTSYD` and `RRPONTSYD` are already in billions. Plotting the
two on a shared axis without this conversion produces a silent three-order-of-
magnitude error, which was present in an earlier revision and is corrected here.

**Rolling tightness z-score.** A 252-observation rolling z-score of the primary
spread, computed on the **business-day subset only**. Computing it across the
full calendar index would give a 252-row window spanning roughly fourteen months
rather than one trading year.

### Validation

The pipeline is checked against a reference point not used in its construction.
Public reporting records a Standing Repo Facility operation of **$29.4 billion
on 2025-10-31**. The independently retrieved `RPONTSYD` value for that date is
**29.400**. This equality is asserted in the code and retained as a regression
test.

Two further consistency checks pass: the two largest repo draws in the sample
fall on 2025-12-31 and 2025-10-31, and both dates rank among the highest
observed values of `SOFR99 − IORB` — the co-movement of facility usage and
funding tightness that the analysis presumes. `WRESBAL_bn` ranges from $2,830B
to $4,276B, with the lower bound reflecting the multi-year low in reserve
balances reached in late 2025.

### Not used

**New York Fed Markets Data API** (`https://markets.newyorkfed.org/api/`) was
evaluated for operation-level repo detail. The `search.json` endpoints returned
HTTP 400 for every date-range query attempted, and record requests above 500
were rejected with no offset parameter available for backward pagination,
leaving historical coverage insufficient for the sample window. FRED's
`RPONTSYD` provides the same daily aggregate with complete history and is used
instead. Submitted-versus-accepted amounts and the operation-method breakdown,
available only through that API, are therefore not used.

**FRED graph CSV export** (`fred.stlouisfed.org/graph/fredgraph.csv`) was
unreachable from the compute environment; the authenticated API at
`api.stlouisfed.org` returns identical content and was substituted.

### Limitations

This dataset characterises **system-level** funding conditions. It contains no
payment-level or institution-level intraday data, because none is publicly
available at that granularity. Its role is therefore to establish that the
problem is real and binding, to identify the stress episodes worth studying, and
to calibrate the simulation environment — not to serve as the estimation sample.
The intraday arrival process for the simulator is calibrated separately from
published Fedwire timing statistics, documented in the following section.
