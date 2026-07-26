"""Diagnostic: how does server utility vary with r, and where does
participation kick in? Helps pick a beta that yields an interior optimum."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mechanism_layer.costs import ClientCosts
from mechanism_layer import payments as pay
from mechanism_layer.mechanisms import fedrsg

cfg = fedrsg.FedRSGConfig(beta=50.0, a_curr=0.5)
c = ClientCosts(gamma=1.5, delta=1.5)  # single representative client

print("--- one client's best response vs r (m=1 linear payment) ---")
P = pay.convex_payment(1, cfg.a_curr)
print(f"{'r':>6} {'a*':>7} {'eps*':>7} {'hat_a':>7} {'u_k':>9} {'payment':>9} {'cost':>8}")
for r in [0, 1, 2, 5, 10, 20, 30, 50]:
    a, e, hat, u = fedrsg.client_best_response(P, c, float(r), cfg)
    pmt = P(r, hat); cost = c.total(a, e)
    print(f"{r:6.1f} {a:7.3f} {e:7.3f} {hat:7.3f} {u:9.3f} {pmt:9.3f} {cost:8.3f}")

print("\n--- server utility vs r for several beta (n=1, m=1) ---")
print(f"{'r':>6}", *[f"beta={b:>5}" for b in [50,100,200,400]])
for r in [0,1,2,5,10,20,30,50]:
    a,e,hat,u = fedrsg.client_best_response(P, c, float(r), cfg)
    pmt = P(r, hat)
    row = []
    for b in [50,100,200,400]:
        Us = b*(hat-cfg.a_curr) - pmt
        row.append(f"{Us:9.2f}")
    print(f"{r:6.1f}", *row)
