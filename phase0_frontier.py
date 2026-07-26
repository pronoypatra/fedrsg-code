"""
Phase-0 de-risk #2: is the DESIGN-LEVER FRONTIER clean and interpretable?

The paper's central thesis: tuning the payment SHAPE (convexity m) and the server
weight (beta) moves the equilibrium along an interpretable quality-vs-cost frontier,
with an identifiable sweet spot.  This must be a visually clean curve or the thesis
needs reshaping BEFORE we build the section.

We check, on the current synthetic oracle:
  (1) sweeping m at fixed beta: does accuracy rise monotonically while
      cost-efficiency (utility-per-buck, bang-per-buck) falls -> a real tradeoff?
  (2) sweeping beta at fixed m: does the operating point slide along quality-vs-cost?
  (3) is there an identifiable sweet spot (max utility-per-buck at interior m)?

Run:  code/.venv/bin/python code/phase0_frontier.py
"""
from __future__ import annotations
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mechanism_layer.costs import ClientCosts
from mechanism_layer import payments as pay
from mechanism_layer import metrics
from mechanism_layer.mechanisms import fedrsg


def make_clients(gammas, deltas, a_curr):
    return [ClientCosts(gamma=float(g), delta=float(d), rho=0.0, a_curr=a_curr,
                        comp="log", priv="linear") for g, d in zip(gammas, deltas)]


def score(P, clients, cfg):
    r_star, out = fedrsg.solve(P, clients, cfg)
    s = metrics.summary(out); s["r"] = r_star
    p = s["total_payment"]
    s["bang_per_buck"] = (s["mean_obs_acc"] / p) if p > 1e-9 else float("nan")
    s["util_per_buck"] = (s["server_utility"] / p) if p > 1e-9 else float("nan")
    return s


def sweep_m(clients, cfg, ms=(1, 2, 3, 4, 5)):
    rows = []
    for m in ms:
        s = score(pay.convex_payment(m, cfg.a_curr), clients, cfg); s["m"] = m
        rows.append(s)
    return rows


def main():
    rng = np.random.default_rng(0)
    n = 10; a_curr = 0.5
    gammas = rng.uniform(0.5, 3.0, n); deltas = rng.uniform(0.5, 3.0, n)
    clients = make_clients(gammas, deltas, a_curr)

    print("### (1) SWEEP m at beta=50  (does shape trade accuracy vs efficiency?) ###")
    cfg = fedrsg.FedRSGConfig(beta=50.0, a_curr=a_curr, r_min=0.0, r_max=50.0)
    rows = sweep_m(clients, cfg)
    print(f"{'m':>2}  {'U^s':>8}  {'acc':>6}  {'pay':>8}  {'BpB':>7}  {'UpB':>7}  {'part':>5}")
    for s in rows:
        print(f"{s['m']:>2}  {s['server_utility']:>8.2f}  {s['mean_obs_acc']:>6.3f}  "
              f"{s['total_payment']:>8.2f}  {s['bang_per_buck']:>7.4f}  "
              f"{s['util_per_buck']:>7.3f}  {s['participation']:>5.2f}")
    accs = [s["mean_obs_acc"] for s in rows]
    mono = all(accs[i] <= accs[i+1] + 1e-9 for i in range(len(accs)-1))
    best_upb = max(range(len(rows)), key=lambda i: (rows[i]["util_per_buck"]
                    if rows[i]["util_per_buck"] == rows[i]["util_per_buck"] else -1e9))
    print(f"  accuracy monotone-increasing in m: {mono}")
    print(f"  sweet spot (max utility-per-buck): m={rows[best_upb]['m']}")

    print("\n### (2) SWEEP beta at fixed m=3  (does operating point slide?) ###")
    print(f"{'beta':>5}  {'U^s':>8}  {'acc':>6}  {'pay':>8}  {'part':>5}")
    for beta in (10, 25, 50, 100, 200):
        cfgb = fedrsg.FedRSGConfig(beta=float(beta), a_curr=a_curr, r_min=0.0, r_max=50.0)
        s = score(pay.convex_payment(3, a_curr), clients, cfgb)
        print(f"{beta:>5}  {s['server_utility']:>8.2f}  {s['mean_obs_acc']:>6.3f}  "
              f"{s['total_payment']:>8.2f}  {s['participation']:>5.2f}")


if __name__ == "__main__":
    main()
