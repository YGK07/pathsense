"""PyTorch models for Functionalities 3 and 4.  *** WRITTEN BUT NOT YET RUN -- see docs/ROADMAP.md tasks M1-M4 ***

Encoder: small 1D-CNN over a (6, T) inertial window -> embedding.
SimCLR : NT-Xent loss on two augmented views (functionality 3).
DANN   : gradient-reversal layer + device classifier (functionality 4).
"""
import torch, torch.nn as nn, torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, c_in=6, emb=64):
        super().__init__()
        def blk(i, o, k): return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU())
        self.net = nn.Sequential(blk(c_in, 32, 7), nn.MaxPool1d(2), blk(32, 64, 5), nn.MaxPool1d(2), blk(64, emb, 3), nn.AdaptiveAvgPool1d(1))
        self.emb = emb

    def forward(self, x):                 # (B, C, T) -> (B, emb)
        return self.net(x).squeeze(-1)


class Projector(nn.Module):
    def __init__(self, emb=64, out=32):
        super().__init__(); self.f = nn.Sequential(nn.Linear(emb, emb), nn.ReLU(), nn.Linear(emb, out))

    def forward(self, z): return self.f(z)


def augment(x, jitter=0.05, scale=(0.8, 1.2), max_shift=10):
    """Time-series augmentations: jitter, per-channel scaling, circular shift. x: (B,C,T)."""
    B, C, T = x.shape
    x = x + jitter * torch.randn_like(x)
    x = x * torch.empty(B, C, 1, device=x.device).uniform_(*scale)
    s = int(torch.randint(-max_shift, max_shift + 1, (1,)))
    return torch.roll(x, s, dims=-1)


def nt_xent(z1, z2, tau=0.2):
    z = F.normalize(torch.cat([z1, z2]), dim=1); n = z1.shape[0]
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, tgt)


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None


def grad_reverse(x, lam=1.0): return _GRL.apply(x, lam)


class DANN(nn.Module):
    def __init__(self, n_classes=3, n_devices=5, emb=64, enc=None):
        super().__init__()
        self.enc = enc or Encoder(emb=emb)
        self.cls = nn.Linear(emb, n_classes)
        self.dom = nn.Sequential(nn.Linear(emb, 32), nn.ReLU(), nn.Linear(32, n_devices))

    def forward(self, x, lam=1.0):
        z = self.enc(x)
        return self.cls(z), self.dom(grad_reverse(z, lam))
