"""
Outcome metrics, computed from an equilibrium profile.

Key design point (the whole reason the comparison is fair): a metric is a
function of the OUTCOME (the profile s and the resulting accuracies/payments/
costs), NOT a property of the mechanism.  So we can score ANY mechanism's
equilibrium under BOTH:
  - server utility  U^s   (FedRSG's objective)
  - p-mean welfare  W_p   (FedBR-BG's objective)
No mechanism is disadvantaged by the choice of yardstick; we report both.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Outcome:
    """Everything needed to score an equilibrium, for any mechanism."""
    obs_acc: np.ndarray      # hat_a_k per client (observed accuracy)
    payments: np.ndarray     # p_k per client (server->client transfer)
    costs: np.ndarray        # C_k per client (net cost incurred)
    global_acc: float        # accuracy of the aggregated global model
    alpha: float = 0.0       # server weight on loss (we use accuracy, so 0 typical)
    beta: float = 1.0        # server weight on avg observed accuracy


def client_utilities(o: Outcome) -> np.ndarray:
    """u_k = payment_k - cost_k  (client net utility)."""
    return o.payments - o.costs


def server_utility(o: Outcome, a_curr: float = 0.5) -> float:
    """U^s = beta * sum(hat_a - a_curr) - sum(payments)  (alpha*loss omitted).
    FedRSG's objective. Two modeling choices, both deliberate:
      (i) reward accuracy IMPROVEMENT over the current model (hat_a - a_curr),
          so universal abstention (hat_a=a_curr) scores 0, not beta*a_curr;
      (ii) SUM (not mean) the per-client contributions, so both the benefit and
          the payment terms scale with n -- more good contributors => more value,
          matching FL intuition. (The mean form perversely made the server less
          willing to pay as n grew.)
    """
    return float(o.beta * np.sum(o.obs_acc - a_curr) - np.sum(o.payments))


def p_mean_welfare(o: Outcome, p: float = 1.0, eps: float = 1e-9) -> float:
    """W_p = ( (1/n) sum_k u_k^p )^{1/p}, for p<=1.  FedBR-BG's objective.
    p=1 average, p->0 Nash (geometric mean), p=-inf egalitarian (min)."""
    u = client_utilities(o)
    u = np.maximum(u, eps)   # p-mean needs positive utilities; IR should give u>=0
    n = len(u)
    if abs(p) < 1e-8:                       # p -> 0 : geometric mean (Nash welfare)
        return float(np.exp(np.mean(np.log(u))))
    if np.isneginf(p):                      # p -> -inf : egalitarian (min)
        return float(np.min(u))
    return float((np.mean(u ** p)) ** (1.0 / p))


def participation(o: Outcome, a_curr: float = 0.5, tol: float = 1e-6) -> float:
    """Fraction of clients that contribute above the trivial level
    (hat_a_k > a_curr). Surfaces free-riding / dropout."""
    return float(np.mean(o.obs_acc > a_curr + tol))


def total_payment(o: Outcome) -> float:
    return float(np.sum(o.payments))


def summary(o: Outcome, p_values=(0.2, 0.4, 0.6, 0.8, 1.0)) -> dict:
    """All headline metrics in one dict, for tables/frontier points.
    Keys use integer-percent tags (welfare_p20 == p=0.2) to avoid float-format
    key drift; welfare_avg is the p=1 utilitarian mean (well-defined always)."""
    u = client_utilities(o)
    d = {
        "server_utility": server_utility(o),
        "global_acc": o.global_acc,
        "mean_obs_acc": float(np.mean(o.obs_acc)),
        "total_payment": total_payment(o),
        "participation": participation(o),
        "min_client_utility": float(np.min(u)),
        "mean_client_utility": float(np.mean(u)),   # = utilitarian welfare (p=1), sign-safe
    }
    for p in p_values:
        d[f"welfare_p{int(round(p*100))}"] = p_mean_welfare(o, p)
    d["welfare_avg"] = float(np.mean(u))            # p=1 utilitarian
    d["nash_welfare"] = p_mean_welfare(o, 0.0)
    d["egalitarian"] = p_mean_welfare(o, float("-inf"))
    return d
