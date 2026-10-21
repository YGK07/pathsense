"""Functionality 3 + 4: contrastive pretraining (SimCLR) and device invariance (DANN).

    pip install -r requirements-ml.txt
    python scripts/train_ssl_dann.py                                   # full comparison -> outputs/ssl_dann.json
    python scripts/train_ssl_dann.py --ssl-epochs 2 --dann-epochs 2    # quick check

For each held-out phone model (leave-one-device-out) the encoder is pretrained once with NT-Xent on unlabelled windows
from the other phones; then, for each labelled fraction, four models are fine-tuned on the SAME labelled windows:
  scratch           random encoder, classifier only
  scratch + DANN    random encoder, gradient-reversal device head
  SSL               pretrained encoder
  SSL + DANN        pretrained encoder + device head
and scored on windows from the held-out phone. Windows come from simulated traces over the real campus graph
(data/osm_vit.json if present, else the fake grid).
"""
import argparse, json, os, sys, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsense.simulate import make_campus_graph, make_traces, add_surface_attributes
from pathsense.osm import load_graph
from pathsense.ml.data import make_windows
from pathsense.ml.models import Encoder, Projector, DANN, augment, nt_xent


def acc(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X)[0].argmax(1) == y).float().mean().item()


def pretrain(Xtr, epochs, seed):
    torch.manual_seed(seed)
    enc, proj = Encoder(), Projector()
    opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), 1e-3)
    loss = torch.tensor(0.0)
    for ep in range(epochs):
        enc.train(); proj.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 128):
            b = Xtr[perm[i:i + 128]]
            if len(b) < 8:
                continue
            loss = nt_xent(proj(enc(augment(b))), proj(enc(augment(b)))); opt.zero_grad(); loss.backward(); opt.step()
        print(f"  ssl epoch {ep + 1}/{epochs} loss {loss.item():.3f}", flush=True)
    return enc


def finetune(Xtr, ytr, dtr, idx, n_dev, epochs, enc_state, use_dann, seed):
    torch.manual_seed(seed)
    m = DANN(n_devices=n_dev, enc=Encoder())
    if enc_state is not None:
        m.enc.load_state_dict(enc_state)
    o = torch.optim.Adam(m.parameters(), 1e-3)
    for ep in range(epochs):
        m.train(); perm = idx[torch.randperm(len(idx))]
        lam = (2 / (1 + np.exp(-10 * ep / max(epochs - 1, 1))) - 1) if use_dann else 0.0
        for i in range(0, len(perm), 64):
            j = perm[i:i + 64]
            if len(j) < 4:
                continue
            lc, ld = m(Xtr[j], lam)
            loss = F.cross_entropy(lc, ytr[j]) + (F.cross_entropy(ld, dtr[j]) if use_dann else 0)
            o.zero_grad(); loss.backward(); o.step()
    return m


def main(a):
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    osm = "data/osm_vit.json"
    G = add_surface_attributes(load_graph(osm), seed=a.seed) if os.path.exists(osm) else make_campus_graph(seed=a.seed)
    tr = make_traces(G, n_traces=a.traces, seed=a.seed + 3)
    X, y, dev, _ = make_windows(tr, G)
    X, y = torch.tensor(X), torch.tensor(y)
    n_dev = int(dev.max()) + 1
    print(f"{len(X)} windows from {a.traces} walks, {n_dev} phone models", flush=True)
    rows = []
    for hold in a.holdouts:
        t0 = time.time()
        te = torch.tensor(dev == hold); trn = ~te
        Xtr, ytr, dtr = X[trn], y[trn], torch.tensor(dev[trn.numpy()])
        majority = (y[te] == torch.mode(ytr).values).float().mean().item()
        print(f"held-out phone {hold}: {int(te.sum())} test windows, majority-class baseline {majority:.3f}", flush=True)
        enc_state = pretrain(Xtr, a.ssl_epochs, a.seed + hold).state_dict()
        g = torch.Generator().manual_seed(a.seed + 100 + hold)
        order = torch.randperm(len(Xtr), generator=g)
        for frac in a.fracs:
            idx = order[: max(int(frac * len(Xtr)), 16)]
            for name, state, dann in (("scratch", None, False), ("scratch+DANN", None, True),
                                      ("SSL", enc_state, False), ("SSL+DANN", enc_state, True)):
                m = finetune(Xtr, ytr, dtr, idx, n_dev, a.dann_epochs, state, dann, a.seed + hold)
                r = dict(holdout=hold, frac=frac, model=name, acc=round(acc(m, X[te], y[te]), 4), majority=round(majority, 4))
                rows.append(r); print(f"  labels {frac:>4}: {name:<13} unseen-phone accuracy {r['acc']:.3f}", flush=True)
        print(f"  ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(config=vars(a), n_windows=len(X), results=rows), open(a.out, "w"), indent=1)
    print("summary (mean over held-out phones):")
    for frac in a.fracs:
        line = "  ".join(f"{name} {np.mean([r['acc'] for r in rows if r['frac'] == frac and r['model'] == name]):.3f}"
                         for name in ("scratch", "scratch+DANN", "SSL", "SSL+DANN"))
        print(f"  labels {frac:>4}: {line}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=int, default=120); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ssl-epochs", type=int, default=15); ap.add_argument("--dann-epochs", type=int, default=15)
    ap.add_argument("--holdouts", type=int, nargs="+", default=[0, 2, 4])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.05, 0.1, 0.25, 1.0])
    ap.add_argument("--out", default="outputs/ssl_dann.json")
    main(ap.parse_args())
