"""
Targeted probe: does A(a, eps) actually RISE with the privacy budget eps?

The smoke DP runs looked flat (eps=1 and eps=8 gave near-identical accuracy). Two
candidate causes: (a) too few epochs for eps to express, or (b) the eps range was
on the plateau -- the accuracy-vs-eps curve is steepest at LOW eps and flattens,
so sampling only {1,8} can miss the rise. This probe settles it on MNIST alone by
training to convergence at a WIDE, low-anchored eps spread.

If accuracy clearly rises across eps here, A(a,eps) is real and the full DP sweep
is worth running (just widen/lower the eps grid). If it stays flat even here, we
rethink the privacy-cost measurement before spending the full run.

Output CSV schema:  target_eps,realized_eps,test_acc

Run (GPU):  python dp_probe.py --dataset mnist --out results/dp_probe_mnist.csv
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
    tr, te, meta = get_dataset(dataset, root, augment=False)
    trl = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0)
    tel = DataLoader(te, batch_size=512, shuffle=False, num_workers=0)
    model = build_model(meta).to(device)
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)   # DP-SGD: no momentum
    engine = PrivacyEngine()
    model, opt, trl = engine.make_private_with_epsilon(
        module=model, optimizer=opt, data_loader=trl, target_epsilon=target_eps,
        target_delta=delta, epochs=epochs, max_grad_norm=max_grad_norm)
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in trl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"  [eps~{target_eps}] epoch {ep}/{epochs}  "
                  f"acc={test_acc(model, tel, device):.4f}  "
                  f"eps_so_far={engine.get_epsilon(delta):.2f}", flush=True)
    return engine.get_epsilon(delta), test_acc(model, tel, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    # WIDE, low-anchored spread -- the rise lives at low eps
    ap.add_argument("--epsilons", type=float, nargs="+",
                    default=[0.2, 0.5, 1, 2, 4, 8, 16])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=512)
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
            print(f"[{args.dataset}] eps={te}  realized={realized:.3f}  acc={acc:.4f}", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
