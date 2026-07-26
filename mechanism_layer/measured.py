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
    """C^comp(a): monotone, shape-preserving interpolation of the measured
    compute-to-reach-a curve, centered so cost(a_curr)=0."""
    def __init__(self, ds: str, a_curr: float = 0.5, normalize: bool = True):
        d = _load(ds)["ccomp"]
        a = np.asarray(d["a_grid"], float)
        c = np.asarray(d["cost_steps"], float)
        # enforce non-decreasing (measurement noise can dip); cumulative max
        c = np.maximum.accumulate(c)
        self.a_min, self.a_max = float(a[0]), float(a[-1])
        self._a, self._c = a, c
        self.a_curr = a_curr
        try:
            from scipy.interpolate import PchipInterpolator
            self._f = PchipInterpolator(a, c, extrapolate=True)
        except Exception:                      # fallback: linear interp
            self._f = lambda x: np.interp(x, a, c)
        # optional normalization: express cost in units where the max is 1 and
        # cost(a_curr)=0, so it is comparable across datasets and centered.
        self._c0 = float(self._f(a_curr)) if a_curr >= self.a_min else 0.0
        self._scale = (float(c[-1]) - self._c0) if normalize else 1.0
        self._scale = self._scale if abs(self._scale) > 1e-9 else 1.0

    def cost(self, a: float) -> float:
        a = float(np.clip(a, self.a_min, self.a_max))
        raw = float(self._f(a)) - self._c0
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
