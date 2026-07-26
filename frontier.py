"""
The money-shot experiment: the (client welfare, server utility) frontier.

Design (per decision): ONE solver, ONE substrate, different PAYMENT RULES.
Every mechanism is FedRSG-EquiSolver run with a different payment function on
IDENTICAL clients / costs / accuracy oracle, so all outcomes are in the SAME
units and both metrics (server utility U^s AND p-mean client welfare) are
directly comparable.  This operationalizes the paper's "FedRSG subsumes them":
  - iFedCrowd  == linear payment (m=1)          [linear functional payment]
  - constant   == fixed-action payment          [forward Stackelberg]
  - FedRSG     == convex payment, m in {1..5}    [optimized payment shape]
  - FedBR-BG   == budget-balanced payment        [Sec: budget-balanced mechanism]

Thesis to test: prior mechanisms are POINTS; FedRSG traces a FRONTIER that
reaches/dominates them, and the gap should be largest under client heterogeneity.

Run:  code/.venv/bin/python code/frontier.py
"""
from __future__ import annotations
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mechanism_layer.costs import ClientCosts
from mechanism_layer import payments as pay
from mechanism_layer import metrics
from mechanism_layer.mechanisms import fedrsg


def make_clients(gammas, deltas, a_curr=0.5):
    return [ClientCosts(gamma=float(g), delta=float(d), rho=0.0, a_curr=a_curr,
                        comp="log", priv="linear") for g, d in zip(gammas, deltas)]


def score(P, clients, cfg):
    """Solve FedRSG with payment P, return (r*, summary-with-r)."""
    r_star, out = fedrsg.solve(P, clients, cfg)
    s = metrics.summary(out)
    s["r"] = r_star
    return s


def run(seed=0, n=10, hetero=True):
    rng = np.random.default_rng(seed)
    if hetero:
        gammas = rng.uniform(0.5, 3.0, n)
        deltas = rng.uniform(0.5, 3.0, n)
    else:
        gammas = np.full(n, 1.5); deltas = np.full(n, 1.5)

    # sum-form server utility: each participating client contributes
    # beta*(hat_a-0.5) for its payment; beta=50 makes participation worthwhile
    # (diag: 1 client at r=20 gives improvement 0.32 for payment 12.8).
    cfg = fedrsg.FedRSGConfig(beta=50.0, a_curr=0.5, r_min=0.0, r_max=50.0)
    clients = make_clients(gammas, deltas, cfg.a_curr)

    res = {"gammas": gammas, "deltas": deltas, "cfg": cfg, "fedrsg": [], "baselines": {}}

    # FedRSG frontier: sweep payment convexity m
    for m in [1, 2, 3, 4, 5]:
        s = score(pay.convex_payment(m, cfg.a_curr), clients, cfg)
        s["m"] = m
        res["fedrsg"].append(s)

    # baselines as payment rules on the SAME substrate
    res["baselines"]["iFedCrowd (linear m=1)"] = score(pay.convex_payment(1, cfg.a_curr), clients, cfg)
    res["baselines"]["constant (fwd-Stackelberg)"] = score(pay.constant_payment(cfg.a_curr), clients, cfg)
    return res


def print_table(res):
    def row(label, d):
        return (f"{label:26s}  U^s={d.get('server_utility',float('nan')):8.2f}  "
                f"Wavg={d.get('welfare_avg',float('nan')):7.3f}  "
                f"minU={d.get('min_client_utility',float('nan')):7.3f}  "
                f"acc={d.get('mean_obs_acc',float('nan')):.3f}  "
                f"pay={d.get('total_payment',float('nan')):8.2f}  "
                f"r={(d.get('r') if d.get('r') is not None else float('nan')):5.2f}  "
                f"part={d.get('participation',float('nan')):.2f}")
    print("\n=== FedRSG frontier (varying payment convexity m) ===")
    for p in res["fedrsg"]:
        print(row(f"FedRSG m={p['m']}", p))
    print("\n=== Baselines (as payment rules, same substrate) ===")
    for label, d in res["baselines"].items():
        print(row(label, d))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6, 5))
    fx = [p["welfare_avg"] for p in res["fedrsg"]]
    fy = [p["server_utility"] for p in res["fedrsg"]]
    ax.plot(fx, fy, "-o", label="FedRSG (vary payment shape $m$)")
    for p in res["fedrsg"]:
        ax.annotate(f"m={p['m']}", (p["welfare_avg"], p["server_utility"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    for (label, d), mk in zip(res["baselines"].items(), ["^", "s", "D", "v"]):
        ax.scatter([d["welfare_avg"]], [d["server_utility"]], marker=mk, s=90, label=label, zorder=5)
    ax.set_xlabel("client welfare  (mean net utility)")
    ax.set_ylabel("server utility  $U^s$")
    ax.set_title("Welfare--Server-utility frontier")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    base = os.path.splitext(path)[0]
    fig.savefig(base + ".pdf"); fig.savefig(base + ".png", dpi=150)
    plt.close(fig)
    print(f"\nsaved {base}.pdf (+ .png)")


if __name__ == "__main__":
    print("### HETEROGENEOUS clients ###")
    res = run(seed=0, n=10, hetero=True)
    print_table(res)
    plot(res, os.path.join(os.path.dirname(__file__), "figures", "frontier_hetero.png"))

    print("\n### HOMOGENEOUS clients (control) ###")
    resh = run(seed=0, n=10, hetero=False)
    print_table(resh)
