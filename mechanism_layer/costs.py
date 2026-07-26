"""
Cost functions for the FedRSG mechanism layer.

All costs are per-client and satisfy the paper's shape assumptions:
  - computation cost C^comp(a): convex, increasing, C^comp(a_curr)=0, ->inf as a->1
  - privacy cost     C^priv(eps): convex, increasing, C^priv(0)=0
  - data-collection  C^coll(F): convex, increasing, C^coll(0)=0
  - communication    C^comm: constant (decision-independent); not modeled here.

These are plain numpy callables so they can be reused by every mechanism
(FedRSG, FedBR-BG, iFedCrowd) and by later fitted-oracle experiments.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# ---------- computation cost C^comp(a) ----------

def comp_cost_log(a: float, a_curr: float = 0.5) -> float:
    """C^comp(a) = log( a_curr(1-a_curr) / (a(1-a)) ).  (paper's Prop-2 form)
    =0 at a=a_curr, ->inf as a->1, convex & increasing on [a_curr,1)."""
    a = np.clip(a, a_curr, 1 - 1e-9)
    return float(np.log((a_curr * (1 - a_curr)) / (a * (1 - a))))


def comp_cost_rational(a: float, a_curr: float = 0.5) -> float:
    """C^comp(a) = a_curr(1-a_curr)/(a(1-a)) - 1.  (config C_1 in the paper)"""
    a = np.clip(a, a_curr, 1 - 1e-9)
    return float((a_curr * (1 - a_curr)) / (a * (1 - a)) - 1.0)


# ---------- privacy cost C^priv(eps) ----------

def priv_cost_linear(eps: float) -> float:
    """C^priv(eps) = eps (convex-weakly; increasing; 0 at 0)."""
    return float(max(eps, 0.0))


def priv_cost_quadratic(eps: float) -> float:
    """C^priv(eps) = eps^2 (strictly convex)."""
    return float(max(eps, 0.0) ** 2)


# ---------- data-collection cost C^coll(F) ----------

def coll_cost_exp(F: float, rho: float = 1.0) -> float:
    """C^coll(F) = exp(rho*F) - 1  (iFedCrowd-style; 0 at F=0, convex)."""
    return float(np.exp(rho * max(F, 0.0)) - 1.0)


def coll_cost_linear(F: float, c: float = 1.0) -> float:
    """C^coll(F) = c*F  (FedBR-BG-style linear data cost)."""
    return float(c * max(F, 0.0))


@dataclass
class ClientCosts:
    """Bundles a client's weighted cost functions and per-client weights.

    net cost = gamma*C^comp(a) + delta*C^priv(eps) + rho*C^coll(F)
    (communication term is a decision-independent constant, kappa=0 here).
    """
    gamma: float = 1.0          # computation weight
    delta: float = 1.0          # privacy weight
    rho: float = 0.0            # data-collection weight (0 => F inactive)
    a_curr: float = 0.5
    comp: str = "log"           # "log" | "rational" | "measured"
    priv: str = "linear"        # "linear" | "quadratic"
    coll: str = "exp"           # "exp" | "linear"
    comp_fn: object = None      # callable C^comp(a); overrides `comp` when set
                                # (used for the MEASURED oracle, measured.MeasuredComp.cost)

    def _comp(self, a: float) -> float:
        if self.comp_fn is not None:
            return float(self.comp_fn(a))
        return comp_cost_log(a, self.a_curr) if self.comp == "log" else comp_cost_rational(a, self.a_curr)

    def _priv(self, eps: float) -> float:
        return priv_cost_linear(eps) if self.priv == "linear" else priv_cost_quadratic(eps)

    def _coll(self, F: float) -> float:
        return coll_cost_exp(F) if self.coll == "exp" else coll_cost_linear(F)

    def total(self, a: float, eps: float = 0.0, F: float = 0.0) -> float:
        return (self.gamma * self._comp(a)
                + self.delta * self._priv(eps)
                + self.rho * self._coll(F))
