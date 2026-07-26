"""
iFedCrowd (Kang et al., AAAI-23): linear-rate Stackelberg mechanism.

Server commits to reward RATES (r1, r2); client best response (their Thm.1,
in the T=T_min regime we compare in, so completion time drops out):
  reward   v_k = r1 * a_k + r2 * F_k        (linear in outcomes)
  cost     C_k = gamma_k (1+a_k) log(1+a_k) + exp(delta_k * F_k) - 1
  utility  u_k = v_k - C_k

Client picks (a_k, F_k) maximizing u_k (concave); server picks (r1,r2)
maximizing its utility  U = (1/n) sum(alpha a_k + beta F_k) - sum(v_k).

This is already a *linear functional payment* -- i.e. an instance of FedRSG
with P linear in the client's outcomes. We include it (a) as a recovery target
(Job 1) and (b) as the m=1 point on the frontier.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

from ..metrics import Outcome


@dataclass
class IFedConfig:
    a_max: float = 5.0        # a_k here is iFedCrowd's "accuracy level" (unbounded-ish)
    F_max: float = 5.0
    r1_max: float = 30.0
    r2_max: float = 15.0
    alpha: float = 80.0       # their system params
    beta: float = 50.0


def _client_best_response(r1, r2, gamma, delta, cfg: IFedConfig):
    """Maximize u_k = r1 a + r2 F - gamma(1+a)log(1+a) - (exp(delta F)-1)."""
    def negu(x):
        a, F = x
        cost = gamma * (1 + a) * np.log(1 + a) + (np.exp(delta * F) - 1)
        return -(r1 * a + r2 * F - cost)
    res = minimize(negu, x0=[0.5, 0.5],
                   bounds=[(0, cfg.a_max), (0, cfg.F_max)], method="L-BFGS-B")
    a, F = res.x
    return float(a), float(F)


def solve(gammas: np.ndarray, deltas: np.ndarray, cfg: IFedConfig) -> tuple[tuple, Outcome]:
    """Find server-optimal (r1*, r2*) and the induced equilibrium.
    Returns ((r1*,r2*), Outcome)."""
    gammas = np.asarray(gammas, float); deltas = np.asarray(deltas, float)
    n = len(gammas)

    def neg_server(r):
        r1, r2 = r
        A = np.zeros(n); F = np.zeros(n)
        for k in range(n):
            A[k], F[k] = _client_best_response(r1, r2, gammas[k], deltas[k], cfg)
        v = r1 * A + r2 * F
        U = np.mean(cfg.alpha * A + cfg.beta * F) - np.sum(v)
        return -U

    res = minimize(neg_server, x0=[cfg.r1_max / 2, cfg.r2_max / 2],
                   bounds=[(0.01, cfg.r1_max), (0.01, cfg.r2_max)], method="L-BFGS-B")
    r1s, r2s = float(res.x[0]), float(res.x[1])

    A = np.zeros(n); F = np.zeros(n)
    for k in range(n):
        A[k], F[k] = _client_best_response(r1s, r2s, gammas[k], deltas[k], cfg)
    v = r1s * A + r2s * F
    costs = gammas * (1 + A) * np.log(1 + A) + (np.exp(deltas * F) - 1)
    # normalize A into [0.5,1]-style observed accuracy for cross-metric scoring
    obs = 0.5 + 0.5 * (A / (A.max() + 1e-9)) if A.max() > 0 else np.full(n, 0.5)
    out = Outcome(obs_acc=obs, payments=v, costs=costs,
                  global_acc=float(np.mean(obs)), alpha=0.0, beta=cfg.beta)
    return (r1s, r2s), out
