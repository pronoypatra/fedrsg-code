"""
Phase-0 de-risk: does the RSG (functional-commitment) payment buy measurably
more than a FORWARD-Stackelberg (fixed-action) payment?  This is THE
make-or-break result -- the empirical justification for the whole RSG framing.

Why the previous attempt collapsed to r*=0
-------------------------------------------
The old "forward" baseline (constant_payment) paid a flat r for ANY improvement
hat_a > a_curr.  A client then crosses the threshold by an epsilon, banks the
full r, and stops -- so paying anything buys ~zero accuracy and the server
rationally sets r*=0.  That is a strawman contract, not a fair forward baseline.

The correct forward-Stackelberg baseline
-----------------------------------------
A forward leader commits to a fixed ACTION: a single take-it-or-leave-it contract
(p, a_target) -- "deliver observed accuracy >= a_target and I pay you exactly p,
else nothing."  The server optimizes over (p, a_target), anticipating the clients'
best response.  This is genuinely forward Stackelberg (one committed action), and
RSG's payment SCHEDULE provably generalizes it (a schedule can place a step at any
single point).  So RSG >= forward by construction; the question is the MAGNITUDE
of the gap and WHERE it appears.

The gap comes from heterogeneity: ONE (p, a_target) contract for all clients is
"one price fits nobody" -- high-cost clients drop out or the server overpays the
cheap ones.  RSG's schedule is a menu each heterogeneous client self-selects along.

Run:  code/.venv/bin/python code/phase0_gap.py
"""
from __future__ import annotations
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mechanism_layer.costs import ClientCosts
from mechanism_layer import payments as pay
from mechanism_layer import metrics
from mechanism_layer.oracle import observed_accuracy
from mechanism_layer.mechanisms import fedrsg


# ----------------------------------------------------------------------------
# Forward-Stackelberg: single fixed-action contract (p, a_target) for ALL clients
# ----------------------------------------------------------------------------

def _min_cost_to_target(costs: ClientCosts, target: float, cfg: fedrsg.FedRSGConfig,
                        A=observed_accuracy, a_res: int = 240) -> tuple[float, float, float]:
    """Cheapest (a, eps) with A(a,eps) >= target for this client.

    Both a and eps raise A and raise cost, so the constraint binds: A(a,eps)=target.
    For the default oracle A=(a*eps+0.5)/(1+eps) this gives eps=(target-0.5)/(a-target)
    for a>target (target>=0.5).  We line-search a over (target, 1) and take the eps
    from the level set; robust and exact for the synthetic oracle.  Returns
    (min_cost, a*, eps*).  If target<=a_curr the trivial action already meets it.
    """
    if target <= cfg.a_curr + 1e-12:
        return costs.total(cfg.a_curr, 0.0), cfg.a_curr, 0.0
    best = (np.inf, cfg.a_curr, 0.0)
    lo = max(target, cfg.a_curr) + 1e-6
    for a in np.linspace(lo, 1 - cfg.eta, a_res):
        # eps on the level set A(a,eps)=target (closed form for default oracle)
        denom = a - target
        if denom <= 0:
            continue
        eps = (target - 0.5) / denom
        if eps < 0 or eps > cfg.eps_max:
            continue
        # verify (guards against a swapped-in oracle) and score
        if A(a, eps) < target - 1e-6:
            continue
        c = costs.total(a, eps)
        if c < best[0]:
            best = (c, a, eps)
    return best


def forward_stackelberg(clients, cfg: fedrsg.FedRSGConfig, A=observed_accuracy,
                        p_grid: int = 80, t_grid: int = 60):
    """Server commits to ONE (p, a_target) contract, anticipating best response.

    Client best response to (p, a_target): reach a_target at min cost if that is
    worth it (p >= min_cost => net p-min_cost >= 0), else abstain (IR).  Since the
    payment is flat above the target, the client never overshoots -- it sits exactly
    at a_target or drops out.  Server maximizes U^s = beta*sum(hat-a_curr) - sum(pay).
    """
    # precompute each client's min cost to each candidate target (independent of p)
    targets = np.linspace(cfg.a_curr + 1e-3, 1 - cfg.eta, t_grid)
    # per-client, per-target: (min_cost, a, eps)
    mc = [[_min_cost_to_target(c, t, cfg, A) for t in targets] for c in clients]

    p_hi = cfg.r_max  # reuse the same payment magnitude budget as RSG's r_max
    p_vals = np.linspace(0.0, p_hi, p_grid)

    best = None
    for ti, t in enumerate(targets):
        for p in p_vals:
            hat = np.empty(len(clients)); paid = np.empty(len(clients)); cst = np.empty(len(clients))
            for k, c in enumerate(clients):
                mincost, a, eps = mc[k][ti]
                if mincost <= p + 1e-12:          # participate: reach target
                    hat[k] = A(a, eps); paid[k] = p; cst[k] = c.total(a, eps)
                else:                              # abstain (individually rational)
                    hat[k] = A(cfg.a_curr, 0.0); paid[k] = 0.0; cst[k] = c.total(cfg.a_curr, 0.0)
            Us = cfg.beta * np.sum(hat - cfg.a_curr) - np.sum(paid)
            if best is None or Us > best["server_utility"]:
                out = metrics.Outcome(obs_acc=hat.copy(), payments=paid.copy(),
                                      costs=cst.copy(), global_acc=float(np.mean(hat)),
                                      alpha=cfg.alpha, beta=cfg.beta)
                s = metrics.summary(out)
                s.update({"p": float(p), "a_target": float(t)})
                best = s
    return best


# ----------------------------------------------------------------------------
# RSG: server picks the best payment SCHEDULE within the convex family {m}
# ----------------------------------------------------------------------------

def rsg_best(clients, cfg: fedrsg.FedRSGConfig, A=observed_accuracy, ms=(1, 2, 3, 4, 5)):
    """RSG server optimizes over payment shapes m and scale r; report the best
    schedule (max server utility) and the whole frontier."""
    frontier = []
    for m in ms:
        r_star, out = fedrsg.solve(pay.convex_payment(m, cfg.a_curr), clients, cfg, A)
        s = metrics.summary(out); s["m"] = m; s["r"] = r_star
        frontier.append(s)
    best = max(frontier, key=lambda s: s["server_utility"])
    return best, frontier


# ----------------------------------------------------------------------------

def make_clients(gammas, deltas, a_curr):
    return [ClientCosts(gamma=float(g), delta=float(d), rho=0.0, a_curr=a_curr,
                        comp="log", priv="linear") for g, d in zip(gammas, deltas)]


def _fmt(tag, s):
    return (f"{tag:24s}  U^s={s['server_utility']:8.2f}  "
            f"Wavg={s['welfare_avg']:7.3f}  minU={s['min_client_utility']:7.3f}  "
            f"acc={s['mean_obs_acc']:.3f}  pay={s['total_payment']:7.2f}  "
            f"part={s['participation']:.2f}")


def run_regime(tag, seed, n, hetero):
    rng = np.random.default_rng(seed)
    a_curr = 0.5
    if hetero:
        gammas = rng.uniform(0.5, 3.0, n); deltas = rng.uniform(0.5, 3.0, n)
    else:
        gammas = np.full(n, 1.5); deltas = np.full(n, 1.5)
    cfg = fedrsg.FedRSGConfig(beta=50.0, a_curr=a_curr, r_min=0.0, r_max=50.0, r_grid=60)
    clients = make_clients(gammas, deltas, a_curr)

    fwd = forward_stackelberg(clients, cfg)
    rsg, frontier = rsg_best(clients, cfg)

    print(f"\n### {tag} (n={n}, seed={seed}) ###")
    print(_fmt("forward-Stackelberg", fwd) + f"   [p={fwd['p']:.2f}, a_target={fwd['a_target']:.3f}]")
    print(_fmt(f"RSG best (m={rsg['m']})", rsg) + f"   [r={rsg['r']:.2f}]")
    gap = rsg["server_utility"] - fwd["server_utility"]
    rel = 100.0 * gap / abs(fwd["server_utility"]) if abs(fwd["server_utility"]) > 1e-9 else float("nan")
    print(f"  --> U^s gap (RSG - forward) = {gap:+.2f}   ({rel:+.1f}%)")
    return {"forward": fwd, "rsg": rsg, "frontier": frontier, "gap": gap, "rel": rel}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="one representative config only")
    args = ap.parse_args()
    if args.fast:
        run_regime("HETEROGENEOUS", seed=0, n=10, hetero=True)
        run_regime("HOMOGENEOUS", seed=0, n=10, hetero=False)
    else:
        for tag, het in [("HETEROGENEOUS", True), ("HOMOGENEOUS", False)]:
            for n in (5, 10, 20):
                run_regime(tag, seed=0, n=n, hetero=het)
