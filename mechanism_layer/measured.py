"""
Measured oracles: drop-in C^comp(a) and A(a,eps) built from the GPU grounding fits
(grounding/fitted/<ds>.json), so the mechanism runs on functions we MEASURED rather
than the synthetic closed forms.  Same call signatures as oracle.observed_accuracy
and costs.comp_cost_*, so existing solver code works unchanged.

  MeasuredComp(ds).cost(a)      -> compute cost (steps), smooth-monotone-interpolated
  MeasuredAccuracy(ds)(a, eps)  -> observed accuracy A(a,eps)

Design notes
------------
C^comp: the fit stores (a_grid, cost_steps).  cost_steps is a staircase (eval was
  every N steps) but the underlying cost is continuous and monotone, so we build a
  monotone, convex-respecting interpolant: enforce non-decreasing, then PCHIP
  (shape-preserving, no overshoot).  Normalized to cost(a_curr)=0 so it plugs into
  the paper's centered utility (client pays only for improvement over a_curr).
A(a,eps): the fit stores the saturating form A_inf - gap*exp(-eps/tau), which is
  concave & increasing in eps.  The measured runs were at the model's own accuracy
  ceiling, so we treat the fitted curve as A(a_max, eps) and scale by (a/a_max) to
  carry the mild dependence on the client's underlying accuracy a (A must be
  increasing in a; linear scaling is the minimal monotone choice and keeps
  A(a,eps)<=a as eps->inf).
"""
from __future__ import annotations
import os, json
import numpy as np

_FITDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "grounding", "fitted")


def _load(ds: str) -> dict:
    with open(os.path.join(_FITDIR, f"{ds}.json")) as fh:
        return json.load(fh)


class MeasuredComp:
    """C^comp(a): a SMOOTH, monotone, convex cost fitted to the measured
    compute-to-reach-a points, centered so cost(a_curr)=0.

    The raw measurement is a staircase (accuracy was checked every N optimizer
    steps), but the underlying cost of accuracy is smooth; a staircase also breaks
    the unimodality the client solver exploits.  We therefore fit the paper's
    closed convex form
        C(a) = k * log( a_curr(1-a_curr) / (a(1-a)) ) + b*(a - a_curr)
    (smooth, increasing, convex on [a_curr,1), zero at a_curr for b chosen so) by
    non-negative least squares to the measured points.  This both removes the
    staircase and directly checks the assumed functional shape (we expose r2_fit).
    Set smooth=False to fall back to the raw monotone PCHIP interpolant.
    """
    def __init__(self, ds: str, a_curr: float = 0.5, normalize: bool = True,
                 smooth: bool = True):
        d = _load(ds)["ccomp"]
        a = np.asarray(d["a_grid"], float)
        c = np.maximum.accumulate(np.asarray(d["cost_steps"], float))  # non-decreasing
        self.a_min, self.a_max = float(a[0]), float(a[-1])
        self.a_curr = a_curr
        self.smooth = smooth
        self.r2_fit = float("nan")

        if smooth:
            # design matrix: [ log-barrier term , linear term ], both centered at a_curr
            def feats(x):
                x = np.clip(x, 1e-6, 1 - 1e-6)
                base = np.log((a_curr * (1 - a_curr)) / (x * (1 - x)))
                return np.stack([base, (x - a_curr)], axis=-1)
            X = feats(a)
            try:
                from scipy.optimize import nnls
                coef, _ = nnls(X, c)             # non-negative -> keeps convex+increasing
            except Exception:
                coef, *_ = np.linalg.lstsq(X, c, rcond=None)
                coef = np.maximum(coef, 0.0)
            self._coef = coef
            pred = X @ coef
            ss = float(np.sum((c - pred) ** 2)); tot = float(np.sum((c - c.mean()) ** 2)) + 1e-12
            self.r2_fit = 1 - ss / tot
            self._feats = feats
            self._f = lambda x: (feats(np.asarray(x, float)) @ coef)
        else:
            try:
                from scipy.interpolate import PchipInterpolator
                self._f = PchipInterpolator(a, c, extrapolate=True)
            except Exception:
                self._f = lambda x: np.interp(x, a, c)

        self._c0 = float(np.atleast_1d(self._f(a_curr))[0]) if a_curr >= self.a_min else 0.0
        cmax = float(np.atleast_1d(self._f(self.a_max))[0])
        self._scale = (cmax - self._c0) if normalize else 1.0
        self._scale = self._scale if abs(self._scale) > 1e-9 else 1.0

    def cost(self, a: float) -> float:
        a = float(np.clip(a, self.a_min, self.a_max))
        raw = float(np.atleast_1d(self._f(a))[0]) - self._c0
        return max(raw / self._scale, 0.0)      # >=0, and 0 at a_curr


class MeasuredAccuracy:
    """A(a,eps) = [A_inf - gap*exp(-eps/tau)] * (a / a_max_dataset).

    Concave & increasing in eps (from the fit); increasing in a (linear scale).
    Callable with the same (a, eps) signature as oracle.observed_accuracy."""
    def __init__(self, ds: str):
        d = _load(ds)["aeps"]
        self.a_inf = float(d["a_inf"]); self.gap = float(d["gap"]); self.tau = float(d["tau"])
        self.r2 = float(d.get("r2", float("nan")))
        # the accuracy the DP curve was measured at (its non-private ceiling proxy)
        self.a_ref = float(self.a_inf) if self.a_inf > 1e-6 else 1.0

    def __call__(self, a: float, eps: float) -> float:
        base = self.a_inf - self.gap * np.exp(-max(eps, 0.0) / max(self.tau, 1e-6))
        return float(np.clip(base * (a / self.a_ref), 0.0, 1.0))


def measured_oracles(ds: str, a_curr: float = 0.5):
    """Convenience: (comp_cost_fn(a), A(a,eps)) for dataset ds, matching the
    synthetic signatures used by the solver."""
    mc = MeasuredComp(ds, a_curr=a_curr)
    ma = MeasuredAccuracy(ds)
    return (lambda a, _ac=a_curr: mc.cost(a)), ma
