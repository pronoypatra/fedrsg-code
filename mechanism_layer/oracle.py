"""
Accuracy / payoff oracles.

Two interchangeable notions of the observed-accuracy map:

1. ObservedAccuracy A(a, eps): FedRSG's per-client map from true accuracy `a`
   and privacy level `eps` to server-observed accuracy `hat_a`. Concave &
   monotone in each argument (paper Property).  A(a,eps)=(a*eps+0.5)/(1+eps).

2. PooledAccuracy a(S): FedBR-BG / iFedCrowd style payoff that depends on the
   TOTAL contribution S = sum of samples (concave, increasing).  Used when the
   mechanism pays on aggregate data rather than per-client accuracy.

Both are pure functions so the same mechanism code runs on a synthetic oracle
now and a data-fitted oracle later (just swap the callable).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# ---------- FedRSG observed-accuracy A(a, eps) ----------

def observed_accuracy(a: float, eps: float) -> float:
    """A(a,eps) = (a*eps + 0.5)/(1+eps).
    eps=0 -> 0.5 (pure noise); eps->inf -> a. Concave, monotone in a and eps."""
    return float((a * eps + 0.5) / (1.0 + eps))


# ---------- pooled accuracy a(S) (FedBR-BG / scaling-law style) ----------

@dataclass
class ScalingLawAccuracy:
    """a(S) = 1 - alpha * S^{-beta}  (Kaplan-style; concave, increasing).
    Depends only on total contributed data S = sum_i s_i.  Matches FedBR-BG's
    'payoff is a function of ||s||_1' assumption."""
    alpha: float = 1.0
    beta: float = 0.5
    a0: float = 1.0     # ceiling accuracy

    def __call__(self, S: float) -> float:
        S = max(S, 1e-9)
        return float(self.a0 - self.alpha * S ** (-self.beta))

    def deriv(self, S: float) -> float:
        S = max(S, 1e-9)
        return float(self.alpha * self.beta * S ** (-self.beta - 1.0))
