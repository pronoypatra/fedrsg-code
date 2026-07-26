"""
Shared data loaders, models, and evaluation for the Phase-1/Phase-3 GPU runs.

One place for: dataset -> (train, test, meta); meta -> model; model -> accuracy.
All models are DP-SGD compatible (no BatchNorm; concrete Linear layers -- no
LazyLinear, which Opacus cannot wrap before its parameters are materialized).

Datasets
  mnist    : torchvision MNIST                     (1x28x28, 10 cls, CNN)
  cifar10  : torchvision CIFAR-10                   (3x32x32, 10 cls, CNN)
  femnist  : torchvision EMNIST 'balanced' as a    (1x28x28, 47 cls, CNN)
             central stand-in for the LEAF FEMNIST benchmark; the federated
             (per-writer) partition is applied in the mechanism layer, not here
             -- grounding only needs the central cost/accuracy curve.
  adult    : UCI Adult via OpenML (tabular)         (108-d, 2 cls, MLP)  [non-vision]
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from PIL import Image


@dataclass
class DataMeta:
    task: str            # "vision" | "tabular"
    n_cls: int
    in_ch: int = 1       # vision
    img: int = 28        # vision: square side
    in_dim: int = 0      # tabular: feature dim


# ---------------------------------------------------------------- datasets

def get_dataset(name: str, root: str = "./data", augment: bool = True):
    """Return (train_ds, test_ds, meta).

    augment=True adds train-time data augmentation where defined (CIFAR crop/flip);
    set augment=False for DP-SGD, where augmentation worsens the already-noisy
    private gradient signal.
    """
    name = name.lower()
    if name == "mnist":
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))])
        tr = datasets.MNIST(root, train=True, download=True, transform=tf)
        te = datasets.MNIST(root, train=False, download=True, transform=tf)
        return tr, te, DataMeta(task="vision", n_cls=10, in_ch=1, img=28)
    if name == "cifar10":
        norm = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2470, 0.2435, 0.2616))
        # augment the TRAIN set (random crop + flip) to curb overfitting -- without
        # this the small CNN peaks early then degrades, giving a non-monotone
        # (unusable) C^comp curve. Test transform stays clean.
        tf_te = transforms.Compose([transforms.ToTensor(), norm])
        tf_tr = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), norm]) if augment else tf_te
        tr = datasets.CIFAR10(root, train=True, download=True, transform=tf_tr)
        te = datasets.CIFAR10(root, train=False, download=True, transform=tf_te)
        return tr, te, DataMeta(task="vision", n_cls=10, in_ch=3, img=32)
    if name in ("femnist", "emnist"):
        # EMNIST source images are transposed (rotated 90 CCW + horizontally
        # flipped) vs. the usual orientation -- a known torchvision quirk. Undo it
        # BEFORE ToTensor so the CNN sees upright characters.
        deskew = transforms.Compose([
            lambda img: img.transpose(Image.TRANSPOSE),   # PIL: swap x/y -> upright
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3332,))])
        tr = datasets.EMNIST(root, split="balanced", train=True, download=True, transform=deskew)
        te = datasets.EMNIST(root, split="balanced", train=False, download=True, transform=deskew)
        return tr, te, DataMeta(task="vision", n_cls=47, in_ch=1, img=28)
    if name == "adult":
        return _get_adult(root)
    raise ValueError(f"unknown dataset {name!r}")


def _get_adult(root: str, n_retries: int = 3):
    """UCI Adult (income >50k) as a binary tabular task, via OpenML.

    OpenML is occasionally flaky (503/504); retry a few times, and if it is truly
    unreachable fall back to a fully-offline tabular set (sklearn digits) so the
    non-vision run never dies on a transient network error.
    """
    try:
        from sklearn.datasets import fetch_openml, load_digits
        from sklearn.preprocessing import StandardScaler
    except Exception as e:                                 # pragma: no cover
        raise SystemExit("adult needs scikit-learn: pip install -r requirements-gpu.txt") from e

    df = None
    for attempt in range(n_retries):
        try:
            ds = fetch_openml("adult", version=2, as_frame=True, data_home=root)
            df = ds.frame.dropna()
            break
        except Exception as ex:                            # network/500/504 etc.
            print(f"[adult] OpenML fetch failed (attempt {attempt+1}/{n_retries}): {ex}", flush=True)

    if df is None:
        print("[adult] OpenML unreachable -> falling back to offline sklearn digits "
              "(tabular 64-d, 10 cls). Report as 'Digits' if used.", flush=True)
        dg = load_digits()
        Xc = StandardScaler().fit_transform(dg.data).astype(np.float32)
        y = dg.target.astype(np.int64)
        return _tab_split(Xc, y, n_cls=10)

    tgt = "class" if "class" in df.columns else df.columns[-1]
    y = df[tgt].astype(str).str.contains(">50").astype(np.int64).to_numpy()
    X = df.drop(columns=[tgt])
    num = X.select_dtypes(include=["number"])
    cat = X.select_dtypes(exclude=["number"])
    Xc = np.asarray(np.concatenate(
        [StandardScaler().fit_transform(num.to_numpy(dtype=np.float64)),
         _onehot(cat)], axis=1), dtype=np.float32)
    return _tab_split(Xc, y, n_cls=2)


def _tab_split(Xc, y, n_cls, frac=0.8):
    n = len(Xc); rng = np.random.default_rng(0); idx = rng.permutation(n); cut = int(frac * n)
    def _ds(i):
        return TensorDataset(torch.from_numpy(Xc[i]), torch.from_numpy(y[i]))
    return _ds(idx[:cut]), _ds(idx[cut:]), DataMeta(task="tabular", n_cls=n_cls, in_dim=Xc.shape[1])


def _onehot(cat_df):
    import pandas as pd
    return pd.get_dummies(cat_df).to_numpy(dtype=np.float64)


# ---------------------------------------------------------------- models

class SmallCNN(nn.Module):
    """Plain CNN, no BatchNorm (DP-compatible). Linear dim is inferred concretely
    (not Lazy) so Opacus can wrap it."""
    def __init__(self, meta: DataMeta):
        super().__init__()
        self.c1 = nn.Conv2d(meta.in_ch, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        with torch.no_grad():
            dummy = torch.zeros(1, meta.in_ch, meta.img, meta.img)
            flat = self._features(dummy).shape[1]
        self.fc1 = nn.Linear(flat, 128)
        self.fc2 = nn.Linear(128, meta.n_cls)

    def _features(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return torch.flatten(x, 1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(self._features(x))))


class MLP(nn.Module):
    """2-hidden-layer MLP for tabular (no BatchNorm; DP-compatible)."""
    def __init__(self, meta: DataMeta):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta.in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, meta.n_cls))

    def forward(self, x):
        return self.net(x)


def build_model(meta: DataMeta) -> nn.Module:
    return SmallCNN(meta) if meta.task == "vision" else MLP(meta)


# ---------------------------------------------------------------- eval

@torch.no_grad()
def test_acc(model, loader, device) -> float:
    model.eval(); correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item(); total += y.numel()
    return correct / total


def device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
