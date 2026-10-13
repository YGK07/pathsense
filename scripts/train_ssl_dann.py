"""Functionality 3 + 4 training script.  *** NOT YET RUN (no torch in the authoring sandbox) -- task M1/M2 ***

    pip install torch
    python scripts/train_ssl_dann.py --ssl-epochs 30 --dann-epochs 30 --holdout-device 4

Experiments to report in the paper (Table 1 rows):
  A. supervised only            B. SSL pretrain -> fine-tune (label fractions 10/25/50/100 %)
  C. DANN, leave-one-device-out (--holdout-device k)  vs  no-DANN baseline
"""
import argparse, os, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsense.simulate import make_campus_graph, make_traces
from pathsense.ml.data import make_windows
from pathsense.ml.models import Encoder, Projector, DANN, augment, nt_xent


def acc(model, X, y, dann=False):
    model.eval()
    with torch.no_grad():
        out = model(X)[0] if dann else model(X)
    return (out.argmax(1) == y).float().mean().item()


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    G = make_campus_graph(seed=a.seed); tr = make_traces(G, n_traces=a.traces, seed=a.seed + 3)
    X, y, dev, _ = make_windows(tr, G)
    X, y = torch.tensor(X), torch.tensor(y)
    te = torch.tensor(dev == a.holdout_device); trn = ~te
    Xtr, ytr, dtr = X[trn], y[trn], torch.tensor(dev[trn.numpy()])
    # --- SSL pretraining on ALL unlabelled windows from training devices ---
    enc, proj = Encoder(), Projector(); opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), 1e-3)
    for ep in range(a.ssl_epochs):
        enc.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 128):
            b = Xtr[perm[i:i + 128]]
            if len(b) < 8: continue
            loss = nt_xent(proj(enc(augment(b))), proj(enc(augment(b)))); opt.zero_grad(); loss.backward(); opt.step()
        print(f"ssl ep{ep} loss {loss.item():.3f}")
    # --- supervised fine-tune with a small labelled fraction, with/without DANN ---
    for frac in (0.1, 0.25, 0.5, 1.0):
        idx = torch.randperm(len(Xtr))[: max(int(frac * len(Xtr)), 16)]
        for use_dann in (False, True):
            m = DANN(n_devices=int(dev.max()) + 1, enc=Encoder()); m.enc.load_state_dict(enc.state_dict())
            o = torch.optim.Adam(m.parameters(), 1e-3)
            for ep in range(a.dann_epochs):
                m.train(); perm = idx[torch.randperm(len(idx))]
                for i in range(0, len(perm), 64):
                    j = perm[i:i + 64]
                    if len(j) < 4: continue
                    p = ep / a.dann_epochs; lam = (2 / (1 + np.exp(-10 * p)) - 1) if use_dann else 0.0
                    lc, ld = m(Xtr[j], lam)
                    loss = F.cross_entropy(lc, ytr[j]) + (F.cross_entropy(ld, dtr[j]) if use_dann else 0)
                    o.zero_grad(); loss.backward(); o.step()
            print(f"label_frac={frac} dann={use_dann} unseen-device acc={acc(m, X[te], y[te], True):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=int, default=120); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ssl-epochs", type=int, default=20); ap.add_argument("--dann-epochs", type=int, default=20)
    ap.add_argument("--holdout-device", type=int, default=4)
    main(ap.parse_args())
