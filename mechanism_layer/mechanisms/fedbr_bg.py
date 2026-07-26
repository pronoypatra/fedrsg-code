"""
FedBR-BG (Murhekar et al., NeurIPS-23): budget-balanced welfare mechanism.

Model here (matched to their paper):
  - each agent i contributes s_i data samples in [0, tau]
  - pooled payoff a_i(s) = a(S), S = sum_i s_i  (concave, via oracle)
  - linear cost c_i(s_i) = c_i * s_i
  - budget-balanced payment  p_i(s) = beta*( c_i(s_i) - (1/(n-1)) sum_{j!=i} c_j(s_j) ),  sum_i p_i = 0
  - agent utility (with payment folded in):
        u_i = a(S) - (1-beta) c_i s_i - (beta/(n-1)) sum_{j!=i} c_j s_j
  - beta* (Def.2): root of  C beta^2 - (A n (n-2) + C) beta + A (n-1)^2 = 0,
        A = (sum_i 1/c_i)^{-1},  C = sum_i c_i;   0 <= beta* <= 1-1/n.

We find the NE by best-response gradient dynamics (their Alg./Thm 3.2).
Then we hand back an Outcome so the SAME metrics (server utility AND p-mean
welfare) can be computed on FedBR-BG's equilibrium -- this is how a
budget-balanced, no-server-utility mechanism still gets placed on our
(welfare, server-utility) frontier plane.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..oracle import ScalingLawAccuracy
from ..metrics import Outcome


def beta_star(c: np.ndarray) -> float:
    """Optimal budget-balance parameter beta* (Def.2)."""
    n = len(c)
    A = 1.0 / np.sum(1.0 / c)
    C = np.sum(c)
    # C b^2 - (A n (n-2) + C) b + A (n-1)^2 = 0
    aa, bb, cc = C, -(A * n * (n - 2) + C), A * (n - 1) ** 2
    disc = bb * bb - 4 * aa * cc
    disc = max(disc, 0.0)
    roots = [(-bb + np.sqrt(disc)) / (2 * aa), (-bb - np.sqrt(disc)) / (2 * aa)]
    # pick the root in [0, 1-1/n]
    ub = 1 - 1.0 / n
    valid = [r for r in roots if -1e-9 <= r <= ub + 1e-9]
    return float(valid[0]) if valid else float(np.clip(min(roots), 0, ub))


@dataclass
class FedBRConfig:
    tau: float = 100.0         # max samples per agent
    beta: float | None = None  # None => use beta* ; 0 => FedBR (no balancing)
    steps: int = 2000
    lr: float = 1.0
    beta_server: float = 1.0   # our server weight, only for scoring


def solve(c: np.ndarray, acc: ScalingLawAccuracy, cfg: FedBRConfig) -> tuple[float, Outcome]:
    """Best-response dynamics to the FedBR-BG Nash equilibrium.

    c: per-agent linear cost coefficients (heterogeneity lives here).
    Returns (beta_used, Outcome).
    """
    c = np.asarray(c, dtype=float)
    n = len(c)
    beta = beta_star(c) if cfg.beta is None else cfg.beta

    s = np.full(n, cfg.tau / 2)  # init
    for _ in range(cfg.steps):
        S = np.sum(s)
        # du_i/ds_i = a'(S) - (1-beta) c_i   (the cross term does not depend on s_i)
        grad = acc.deriv(S) - (1 - beta) * c
        s = np.clip(s + cfg.lr * grad, 0.0, cfg.tau)

    S = np.sum(s)
    # payments (budget-balanced): p_i = beta ( c_i s_i - (1/(n-1)) sum_{j!=i} c_j s_j )
    cost_i = c * s
    mean_others = (np.sum(cost_i) - cost_i) / (n - 1)
    payments = beta * (cost_i - mean_others)

    # per-agent observed accuracy = the shared pooled accuracy a(S)
    ga = acc(S)
    obs = np.full(n, ga)
    out = Outcome(obs_acc=obs, payments=payments, costs=cost_i,
                  global_acc=ga, alpha=0.0, beta=cfg.beta_server)
    return beta, out
