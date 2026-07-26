"""
The 'why reverse Stackelberg' experiment: SAME data-contribution setting,
solved under two solution concepts.

Setting (FedBR-style, Murhekar NeurIPS-23):
  - n agents each choose data contribution s_i in [0, tau]
  - agents VALUE THE MODEL: payoff a(S), S = sum s_i, concave & increasing
    (this is the lambda>0 model-benefit term; FedBR clients are prosumers)
  - linear cost c_i * s_i

Three regimes on this ONE setting:
  (A) FedBR (Nash, no mechanism, beta=0): simultaneous game, best-response
      dynamics -> Nash equilibrium. Exhibits free-riding (agents under-contribute
      because they ignore the externality their data provides to others).
  (B) FedBR-BG (Nash + budget-balanced payment, beta=beta*): their mechanism;
      redistributes cost, still a Nash equilibrium, Sum p = 0.
  (C) FedRSG (reverse Stackelberg): the SERVER commits to a payment FUNCTION of
      contributed data and ANTICIPATES the clients' best response (leader-first).
      This is the only regime with commitment power.

Claim tested: the reverse-Stackelberg leader's COMMITMENT lets the server steer
contributions toward higher total data / model accuracy than the Nash regimes,
because it can pay contingently on what clients deliver. Isolates commitment as
the RSG advantage (same setting, only the solution concept differs).

Run: code/.venv/bin/python code/contrast_nash_vs_rsg.py
"""
from __future__ import annotations
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mechanism_layer.oracle import ScalingLawAccuracy
from mechanism_layer.mechanisms import fedbr_bg


def total_welfare(a_of_S, s, c):
    """Sum of agent net utilities u_i = a(S) - c_i s_i (model-benefit form)."""
    S = np.sum(s)
    return float(np.sum(a_of_S(S) - c * s))


# ---------- (A)/(B) Nash regimes: reuse fedbr_bg best-response ----------

def nash_regime(c, acc, beta):
    cfg = fedbr_bg.FedBRConfig(tau=100.0, beta=beta, steps=4000, lr=1.0, beta_server=1.0)
    b_used, out = fedbr_bg.solve(c, acc, cfg)
    # recover s from cost_i = c*s
    s = out.costs / c
    return {"beta": b_used, "S": float(np.sum(s)), "acc": out.global_acc,
            "welfare": total_welfare(acc, s, c), "s": s}


# ---------- (C) reverse-Stackelberg leader on the SAME data game ----------

def rsg_regime(c, acc, tau=100.0, r_grid=400, r_max=None):
    """Server commits to a per-unit-data payment rate r (pays r per contributed
    sample, i.e. payment function p_i = r * s_i -- a functional commitment on the
    data dimension), ANTICIPATING client best response, and maximizes its own
    utility U^s = beta_s * (a(S) improvement) - total payment.

    Client i best response to rate r: maximize a(S) + r*s_i - c_i*s_i over s_i.
    Since a depends on total S, we solve the induced game for each r (clients
    best-respond simultaneously to r; this inner game is itself concave), then
    the server (leader) picks the r maximizing its utility -- backward induction.
    """
    c = np.asarray(c, float); n = len(c)
    beta_s = 1.0
    if r_max is None:
        r_max = float(np.max(c) * 1.5 + 1.0)

    def inner_equilibrium(r):
        # each client picks s_i to maximize a(S) + (r - c_i) s_i.
        # best-response gradient: da/dS + (r - c_i). Solve the fixed point.
        s = np.full(n, tau / 2)
        for _ in range(4000):
            S = np.sum(s)
            grad = acc.deriv(S) + (r - c)   # d u_i / d s_i
            s = np.clip(s + 1.0 * grad, 0.0, tau)
        return s

    best = None
    for r in np.linspace(0.0, r_max, r_grid):
        s = inner_equilibrium(r)
        S = np.sum(s)
        payment = r * np.sum(s)                       # total paid by server
        # server values accuracy improvement over the a(0) baseline, minus payment
        Us = beta_s * (acc(S) - acc(1e-9)) * n - payment
        if best is None or Us > best["Us"]:
            best = {"r": float(r), "S": float(S), "acc": acc(S),
                    "Us": float(Us), "payment": float(payment),
                    "welfare": total_welfare(acc, s, c), "s": s}
    return best


def run(seed=0, n=10, hetero=True):
    rng = np.random.default_rng(seed)
    c = rng.uniform(0.001, 0.02, n) if hetero else np.full(n, 0.005)
    acc = ScalingLawAccuracy(alpha=1.0, beta=0.5, a0=1.0)

    A = nash_regime(c, acc, beta=0.0)          # (A) FedBR, no mechanism (pure Nash)
    B = nash_regime(c, acc, beta=None)         # (B) FedBR-BG, budget-balanced Nash
    C = rsg_regime(c, acc)                     # (C) FedRSG, reverse-Stackelberg leader
    return c, A, B, C


def report(tag, c, A, B, C):
    print(f"\n### {tag} (n={len(c)}) ###")
    print(f"{'regime':<28}{'total data S':>13}{'model acc':>11}{'welfare':>10}")
    print(f"{'(A) FedBR  (Nash, no mech)':<28}{A['S']:>13.1f}{A['acc']:>11.3f}{A['welfare']:>10.3f}")
    print(f"{'(B) FedBR-BG (Nash, bal.)':<28}{B['S']:>13.1f}{B['acc']:>11.3f}{B['welfare']:>10.3f}")
    print(f"{'(C) FedRSG (rev-Stackelbg)':<28}{C['S']:>13.1f}{C['acc']:>11.3f}{C['welfare']:>10.3f}")
    print(f"    [FedRSG server rate r*={C['r']:.4f}, total payment={C['payment']:.2f}]")


if __name__ == "__main__":
    for tag, het in [("HETEROGENEOUS", True), ("HOMOGENEOUS", False)]:
        c, A, B, C = run(seed=0, n=10, hetero=het)
        report(tag, c, A, B, C)
