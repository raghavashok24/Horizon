"""
Action-conditional intraday liquidity risk certifier.

Given a bank's mid-morning observation, its reserve level, and its PLANNED release
schedule, the certifier predicts end-of-day unsettled-value risk and returns a
conformalized 90% predictive interval (CQR: Romano, Patterson & Candes, NeurIPS 2019).

Deployment shape: a treasury-desk morning gauge. Inputs are all observable at the
decision point; shocks occur post-decision by construction, so certification is a
genuine prediction, not a replay.

Fairness notes:
  * all splits are grouped by simulation seed (shared bank populations otherwise leak)
  * the conformal calibration set is disjoint from both train and test
  * the oracle benchmark is dimension-matched (8 principal components)
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as GBR

TARGET = 0.10          # unsettled-value risk target certified against


def features(D):
    A4 = np.c_[D["A"].mean(1), D["A"][:, 0], D["A"][:, -1], D["A"].std(1)]
    return np.column_stack([D["H"], D["OWN"], D["MULT"][:, None], A4])


class Certifier:
    def __init__(self, alpha: float = 0.10, random_state: int = 0):
        self.alpha = alpha; self.rs = random_state
        self.point = None; self.qlo = None; self.qhi = None; self.qhat = 0.0

    def fit(self, X, y, calib_mask):
        tr = ~calib_mask
        self.point = GBR(max_iter=400, random_state=self.rs).fit(X[tr], y[tr])
        self.qlo = GBR(max_iter=300, loss="quantile", quantile=self.alpha / 2,
                       random_state=self.rs).fit(X[tr], y[tr])
        self.qhi = GBR(max_iter=300, loss="quantile", quantile=1 - self.alpha / 2,
                       random_state=self.rs).fit(X[tr], y[tr])
        lo, hi = self.qlo.predict(X[calib_mask]), self.qhi.predict(X[calib_mask])
        scores = np.maximum(lo - y[calib_mask], y[calib_mask] - hi)   # CQR scores
        n = len(scores)
        self.qhat = float(np.quantile(scores, min(1.0, (1 - self.alpha) * (n + 1) / n)))
        return self

    def certify(self, X):
        """Point risk, conformal interval, and breach flag at the TARGET."""
        p = self.point.predict(X)
        lo = self.qlo.predict(X) - self.qhat
        hi = self.qhi.predict(X) + self.qhat
        return {"risk": p, "lo": lo, "hi": hi, "breach": p > TARGET,
                "breach_possible": hi > TARGET}

    def select_schedule(self, obs_row, mult, candidates):
        """Among candidate release paths, return the one with lowest certified risk.
        In this environment selection recovers release-on-arrival, consistent with
        the analytic liquidity-recycling prediction — reported as validation."""
        feats = np.array([np.r_[obs_row, [mult], [c.mean(), c[0], c[-1], c.std()]]
                          for c in candidates])
        return int(np.argmin(self.point.predict(feats)))
