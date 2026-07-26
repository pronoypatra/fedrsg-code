"""
Phase-1 grounding: measure the COMPUTATION-COST oracle C^comp(a).

Trains a standard model and logs (epoch, test_accuracy).  We invert this locally
to "epochs (compute) needed to reach accuracy a" -- the true, measured cost of
buying accuracy -- and check it is convex & increasing with a sharp rise near the
ceiling (the shape the paper assumes for C^comp).

Output CSV schema:  epoch,test_acc

Run (on GPU server):
  python comp_cost.py --dataset mnist   --epochs 30 --out results/comp_mnist.csv
  python comp_cost.py --dataset cifar10 --epochs 60 --out results/comp_cifar10.csv
  python comp_cost.py --dataset femnist --epochs 40 --out results/comp_femnist.csv
  python comp_cost.py --dataset adult   --epochs 40 --out results/comp_adult.csv
"""
from __future__ import annotations
import argparse, os, csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import get_dataset, build_model, test_acc, device_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--root", default="./data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = device_str()
    tr, te, meta = get_dataset(args.dataset, args.root)
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=2)
    tel = DataLoader(te, batch_size=512, shuffle=False, num_workers=2)

    model = build_model(meta).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["epoch", "test_acc"])
        w.writerow([0, test_acc(model, tel, device)]); fh.flush()
        for ep in range(1, args.epochs + 1):
            model.train()
            for x, y in trl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
            a = test_acc(model, tel, device)
            w.writerow([ep, a]); fh.flush()
            print(f"[{args.dataset}] epoch {ep:3d}  test_acc {a:.4f}", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
