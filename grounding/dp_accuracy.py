"""
Phase-1 grounding: measure the OBSERVED-ACCURACY oracle A(a, eps) under REAL DP.

Trains with DP-SGD (Opacus) at several target privacy budgets eps, records the
*realized* eps from the privacy accountant and the test accuracy.  This is the
empirical A(a, eps): accuracy rising and concave as the privacy budget grows
(eps up -> less noise -> higher accuracy).  A real accountant (not a simulated A)
is the direct answer to "DP is never actually implemented."

Output CSV schema:  target_eps,realized_eps,test_acc

Run (on GPU server):
  python dp_accuracy.py --dataset mnist   --epsilons 0.5 1 2 4 8 --out results/dp_mnist.csv
  python dp_accuracy.py --dataset cifar10 --epsilons 0.5 1 2 4 8 --out results/dp_cifar10.csv
  python dp_accuracy.py --dataset femnist --epsilons 0.5 1 2 4 8 --out results/dp_femnist.csv
  python dp_accuracy.py --dataset adult   --epsilons 0.5 1 2 4 8 --out results/dp_adult.csv

The non-private ceiling (eps -> inf) is the final accuracy from comp_cost.py; we
fit A(a,eps) -> a as eps -> inf against that.
"""
from __future__ import annotations
import argparse, os, csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import get_dataset, build_model, test_acc, device_str

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
except Exception as e:                                 # pragma: no cover
    raise SystemExit("opacus required: pip install -r requirements-gpu.txt") from e


def train_dp(dataset, target_eps, epochs, batch, lr, root, delta, max_grad_norm):
    device = device_str()
    tr, te, meta = get_dataset(dataset, root)
    trl = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=2)
    tel = DataLoader(te, batch_size=512, shuffle=False, num_workers=2)

    model = build_model(meta).to(device)
    model = ModuleValidator.fix(model)                 # ensure DP-compatible layers
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    engine = PrivacyEngine()
    model, opt, trl = engine.make_private_with_epsilon(
        module=model, optimizer=opt, data_loader=trl,
        target_epsilon=target_eps, target_delta=delta,
        epochs=epochs, max_grad_norm=max_grad_norm)

    for ep in range(1, epochs + 1):
        model.train()
        for x, y in trl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        print(f"  [{dataset} eps~{target_eps}] epoch {ep}/{epochs}", flush=True)

    return engine.get_epsilon(delta), test_acc(model, tel, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epsilons", type=float, nargs="+", default=[0.5, 1, 2, 4, 8])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--root", default="./data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["target_eps", "realized_eps", "test_acc"])
        for te in args.epsilons:
            realized, acc = train_dp(args.dataset, te, args.epochs, args.batch,
                                     args.lr, args.root, args.delta, args.max_grad_norm)
            w.writerow([te, realized, acc]); fh.flush()
            print(f"[{args.dataset}] target_eps={te}  realized_eps={realized:.3f}  "
                  f"test_acc={acc:.4f}", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
