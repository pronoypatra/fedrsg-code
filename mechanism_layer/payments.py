"""
Payment-function family for FedRSG and its ablations.

All are P(r, hat_a) with a_curr fixed; centered so P(r, a_curr)=0 and P>=0 for
hat_a>=a_curr (Property in the paper => individual rationality).

  - convex_payment(m):  P = r [ (hat_a/a_curr)^m - 1 ]   (m=1 linear ... m=5 quintic)
  - linear_payment:     m=1 special case (~ iFedCrowd's linear rate)
  - constant_payment:   P = r * 1{hat_a > a_curr}         (forward-Stackelberg
                        fixed action: pay a flat r for any improvement, shape-free)

The three cover the internal ablation: constant (fixed action) < linear (m=1)
< full FedRSG (optimized m). Feeding any of these to fedrsg.solve() runs that
restriction with identical everything-else.
"""
from __future__ import annotations
from typing import Callable


def convex_payment(m: float, a_curr: float = 0.5) -> Callable[[float, float], float]:
    def P(r: float, hat_a: float) -> float:
        if hat_a <= a_curr:
            return 0.0
        return r * ((hat_a / a_curr) ** m - 1.0)
    return P


def linear_payment(a_curr: float = 0.5) -> Callable[[float, float], float]:
    return convex_payment(1.0, a_curr)


def constant_payment(a_curr: float = 0.5) -> Callable[[float, float], float]:
    """Forward-Stackelberg fixed action: flat reward r for any improvement,
    independent of how much hat_a exceeds a_curr (no shape)."""
    def P(r: float, hat_a: float) -> float:
        return r if hat_a > a_curr else 0.0
    return P
