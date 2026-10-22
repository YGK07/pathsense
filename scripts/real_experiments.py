"""Functionalities 1-5 and 7 on the REAL data collected so far (data/raw walks + the real path map), plus the
comparison with a Western-centric routing profile. Nothing here uses the simulator.

    python scripts/real_experiments.py            # -> outputs/real_results.json (about 20-25 minutes on a laptop CPU)
    python scripts/real_experiments.py --quick    # fewer seeds / epochs, for a smoke test

Ground truth: the kerbs walkers marked with the Annotation button (synced walks only), and the path surfaces from
OpenStreetMap tags, the satellite check and team edits (pathsense/conditions.py).

F1  Inertial inference on real walks: roughness (5 Hz high-passed vertical acceleration) per segment against the
    known surface, and slope (forward tilt of the phone) checked for consistency between walks in opposite directions.
F2  Fusion. Each walk is an annotator: it "says" a segment has a kerb if the jolt detector fired there. Single walk,
    majority vote, Dawid-Skene, and both weighted by trace quality (GPS accuracy); do the estimated reliabilities match
    each walk's real accuracy?
F3  SimCLR pretraining on unlabelled windows from all walks, fine-tuned to tell kerb-step windows from walking.
F4  Train on one phone, test on another, with and without a DANN device head.
F5  Hide segments' real labels and predict them from neighbours (surface class, kerb presence).
F7  Pre-processing re-derived from the pilot data: jolt-detector filter cut-off / threshold / spacing chosen by
    leave-one-walk-out, against a generic human-activity-recognition setting; window length for the learned detector.
W   Western-centric profile: a wheelchair router that assumes kerb ramps everywhere and paved surfaces unless tagged
    (what a map built from OSM alone believes), scored on the real VIT map against PathSense.
"""
import argparse, json, math, os, sys, time
from collections import Counter, defaultdict
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import spearmanr, mannwhitneyu
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "app"))
from pathsense.osm import load_graph
from pathsense import conditions as CND
from pathsense.fusion import dawid_skene, majority_vote
from pathsense.features import highpass
from pathsense.io import (load_recording, vertical, detect_events, walk_bounds, align_annotations, edge_index,
                          match_points, walking_window)
from process_walks import find_walks

FS = 50.0
WIN = 64                              # 1.28 s: the window length F7 picked from the pilot walks (2.56 s HAR default)
DEFAULT_LAG = -0.6                    # protocol: the press comes ~0.4-0.9 s after the step


def bal_acc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(np.mean([np.mean(p[y == k] == k) for k in np.unique(y)]))


def prf(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp = (y & p).sum(); fp = (~y & p).sum(); fn = (y & ~p).sum()
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return dict(acc=round(float((y == p).mean()), 3), precision=round(float(pr), 3), recall=round(float(rc), 3),
                f1=round(float(2 * pr * rc / max(pr + rc, 1e-9)), 3), balanced_acc=round(bal_acc(y, p), 3))


def r3(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 3)


def load_world(root):
    G = load_graph(os.path.join(root, "data", "osm_vit.json"))
    for k, (*_, d) in enumerate(G.edges(data=True)):
        d["idx"] = k
    real = json.load(open(os.path.join(root, "outputs", "real_walks.json")))
    team = CND.load(os.path.join(root, "data", "path_conditions.json"), {})
    sat = CND.load(os.path.join(root, "data", "satellite_review.json"), {})
    cond, spots, _ = CND.build(G, real, team, sat)
    recs = {}
    for path in find_walks(os.path.join(root, "data", "raw")):
        r = load_recording(path)
        if any(w["id"] == r.name for w in real["walks"]):
            recs[r.name] = r
    return G, real, cond, spots, recs


def press_lag(w, rule_t):
    """Typical time from a button press to the jolt it marks, for one walk (median of the nearest jolt within 3 s).
    Contributor c01 presses just after the step (lag ~ -0.2 s); c03 presses before it (lag ~ +1 s)."""
    lags = []
    for p in w["labelled"]:
        d = np.asarray(rule_t) - p["t"]; d = d[np.abs(d) <= 3]
        if len(d):
            lags.append(d[np.argmin(np.abs(d))])
    return (float(np.median(lags)), len(lags)) if len(lags) >= 5 else (DEFAULT_LAG, len(lags))


# ----------------------------------------------------------------------------------------------- F1 inertial inference
def passes(G, real, recs):
    """Every time a walk goes along a segment: (walk, edge, t0, t1, direction +1 = along the edge geometry, -1 against)."""
    ix = edge_index(G); geom = {d["idx"]: d["geom"] for *_, d in G.edges(data=True)}
    out = []
    for w in real["walks"]:
        r = recs[w["id"]]; a, b = walk_bounds(r, align_annotations(r)[0])
        g = r.gps[(r.gps[:, 3] <= 30) & (r.gps[:, 0] >= a) & (r.gps[:, 0] <= b)]
        if len(g) < 5:
            continue
        e, _ = match_points(ix, g[:, 1], g[:, 2])
        i = 0
        while i < len(e):
            j = i
            while j + 1 < len(e) and e[j + 1] == e[i]:
                j += 1
            if e[i] >= 0 and j - i >= 2 and g[j, 0] - g[i, 0] >= 6:
                gm = geom[int(e[i])]
                d0 = lambda p: (p[0] - gm[0][0]) ** 2 + (p[1] - gm[0][1]) ** 2
                direction = 1 if d0(g[j, 1:3]) > d0(g[i, 1:3]) else -1
                out.append(dict(walk=w["id"], edge=int(e[i]), t0=float(g[i, 0]), t1=float(g[j, 0]), dir=direction))
            i = j + 1
    return out


def f1_inertial(G, real, cond, recs):
    P = passes(G, real, recs)
    feats = {}
    for w in real["walks"]:
        r = recs[w["id"]]
        hp = highpass(vertical(r), 5.0, r.fs)
        gr = r.grav / np.linalg.norm(r.grav, axis=1, keepdims=True)
        pitch = np.degrees(np.arctan2(gr[:, 1], np.sqrt(gr[:, 0] ** 2 + gr[:, 2] ** 2)))   # tilt of the top edge
        feats[w["id"]] = (r.t, hp, pitch)
    rows = []
    for p in P:
        t, hp, pitch = feats[p["walk"]]
        m = (t >= p["t0"]) & (t <= p["t1"])
        if m.sum() < 100:
            continue
        x = hp[m]
        rows.append(dict(p, rough=float(1.4826 * np.median(np.abs(x - np.median(x)))), pitch=float(np.median(pitch[m]))))
    by_walk = defaultdict(list)
    for x in rows:
        by_walk[x["walk"]].append(x)
    for xs in by_walk.values():                     # remove each walk's own level (person, phone, how it is held)
        mr = np.median([x["rough"] for x in xs]); mp = np.median([x["pitch"] for x in xs])
        for x in xs:
            x["rough_n"] = x["rough"] / mr; x["slope_uv"] = x["dir"] * (x["pitch"] - mp)
    out = dict(passes=len(rows), segments=len({x["edge"] for x in rows}))
    # roughness vs known surface (real sources only)
    rank = {"smooth": 0, "paved": 1, "broken": 2, "unpaved": 2}
    seg_r = defaultdict(list)
    for x in rows:
        seg_r[x["edge"]].append(x["rough_n"])
    ks = [e for e in seg_r if cond[e]["surface_src"] in ("osm", "satellite", "team") and cond[e]["surface"] in rank]
    R = np.array([np.median(seg_r[e]) for e in ks]); C = np.array([rank[cond[e]["surface"]] for e in ks])
    by_cls = {name: [float(v) for v in R[C == k]] for k, name in ((0, "smooth"), (1, "paved"), (2, "broken/unpaved"))}
    sm, pv = by_cls["smooth"], by_cls["paved"]
    out["roughness"] = dict(segments_with_known_surface=len(ks), counts={k: len(v) for k, v in by_cls.items()},
                            median_by_surface={k: r3(np.median(v)) if v else None for k, v in by_cls.items()},
                            spearman_vs_surface_rank=r3(spearmanr(R, C).correlation) if len(ks) > 3 else None,
                            smooth_vs_paved_mannwhitney_p=r3(mannwhitneyu(sm, pv).pvalue) if sm and pv else None)
    # slope: two passes over the same segment by different walks should agree (after turning direction into a sign)
    seg_s = defaultdict(list)
    for x in rows:
        seg_s[x["edge"]].append(x)
    same, opp = [], []
    for xs in seg_s.values():
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                if xs[i]["walk"] != xs[j]["walk"]:
                    (same if xs[i]["dir"] == xs[j]["dir"] else opp).append((xs[i]["slope_uv"], xs[j]["slope_uv"]))
    def agree(pairs):
        if len(pairs) < 4:
            return dict(pairs=len(pairs))
        a, b = np.array(pairs).T
        return dict(pairs=len(pairs), correlation=r3(np.corrcoef(a, b)[0, 1]),
                    sign_agreement=r3(np.mean(np.sign(a) == np.sign(b))))
    out["slope"] = dict(same_direction=agree(same), opposite_direction=agree(opp),
                        typical_offset_deg=r3(np.median([abs(x["slope_uv"]) for x in rows])))
    out["steps"] = "not evaluated: no synced walk has stairs marked yet (double presses), so there is no stairs ground truth"
    return out


# ----------------------------------------------------------------------------------------------- F2 fusion
def f2_fusion(G, real, spots):
    seg = CND.Segments(G)
    foot = {d["idx"] for *_, d in G.edges(data=True) if d.get("osm_highway") not in CND.ROADS}
    walks = real["walks"]
    truth_edges = {e for w in walks if w["labelled"] for e in w["edges"]}
    kerb_edges = {s["edge"] for s in spots if "edge" in s}
    items, annot, labels, wts = [], [], [], []
    for a, w in enumerate(walks):
        hit = set()
        for ev in w["events"]:
            e = seg.nearest(ev["lat"], ev["lon"], CND.KERB_SNAP_M, prefer=foot)
            if e is not None:
                hit.add(e)
        q = float(np.clip(10.0 / max(w["gps_median_acc_m"], 1.0), 0.2, 1.0))      # trace quality from GPS accuracy
        for e in w["edges"]:
            items.append(e); annot.append(a); labels.append(int(e in hit)); wts.append(q)
    items, annot, labels, wts = map(np.array, (items, annot, labels, wts))
    uniq = sorted(set(items.tolist())); pos = {e: i for i, e in enumerate(uniq)}
    it = np.array([pos[e] for e in items])
    T, CM, prior = dawid_skene(it, annot, labels, len(uniq), len(walks), 2)
    Tw, _, _ = dawid_skene(it, annot, labels, len(uniq), len(walks), 2, weights=wts)
    MV, MVw = majority_vote(it, labels, len(uniq), 2), majority_vote(it, labels, len(uniq), 2, wts)
    n_obs = Counter(items.tolist())
    out = dict(n_walks=len(walks), n_observations=int(len(items)), segments_walked=len(uniq),
               segments_with_truth=len(truth_edges & set(uniq)), truth_kerb_segments=len(kerb_edges & truth_edges),
               quality_weights={w["title"]: r3(np.clip(10.0 / max(w["gps_median_acc_m"], 1.0), 0.2, 1.0)) for w in walks})
    for name, keep in (("all segments with a true answer", lambda e: True),
                       ("segments walked 2+ times", lambda e: n_obs[e] >= 2),
                       ("segments walked 3+ times", lambda e: n_obs[e] >= 3)):
        E = [e for e in uniq if e in truth_edges and keep(e)]
        if not E:
            continue
        y = [e in kerb_edges for e in E]; m = np.isin(items, E)
        out[name] = dict(n=len(E), kerb_segments=int(sum(y)),
                         single_walk=prf(np.isin(items[m], list(kerb_edges)), labels[m].astype(bool)),
                         majority_vote=prf(y, [MV[pos[e], 1] > MV[pos[e], 0] for e in E]),
                         majority_vote_quality_weighted=prf(y, [MVw[pos[e], 1] > MVw[pos[e], 0] for e in E]),
                         any_walk=prf(y, [MV[pos[e], 1] > 0 for e in E]),
                         dawid_skene=prf(y, [T[pos[e], 1] > 0.5 for e in E]),
                         dawid_skene_quality_weighted=prf(y, [Tw[pos[e], 1] > 0.5 for e in E]))
    rel, true_acc, who = [], [], []
    for a, w in enumerate(walks):
        m = (annot == a) & np.isin(items, list(truth_edges))
        if m.sum() < 5:
            continue
        y = np.isin(items[m], list(kerb_edges)); p = labels[m].astype(bool)
        rel.append(float((CM[a, [0, 1], [0, 1]] * prior).sum())); true_acc.append(bal_acc(y, p)); who.append(w["title"])
    rho = spearmanr(rel, true_acc).correlation if len(rel) >= 3 else None
    out["reliability"] = dict(walks=[dict(walk=t, estimated=round(r, 3), real_balanced_acc=round(x, 3))
                                     for t, r, x in zip(who, rel, true_acc)], spearman=r3(rho))
    return out


# ----------------------------------------------------------------------------------------------- windows
def channels(r):
    """6 orientation-free channels at 50 Hz: vertical acc, horizontal acc, |acc|, |gyro|, yaw rate, horizontal gyro."""
    t = np.arange(r.t[0], r.t[-1], 1 / FS)
    acc = np.stack([np.interp(t, r.t, r.acc[:, i]) for i in range(3)], 1)
    grav = np.stack([np.interp(t, r.t, r.grav[:, i]) for i in range(3)], 1)
    gyro = np.stack([np.interp(t, r.t, r.gyro[:, i]) for i in range(3)], 1)
    g = grav / np.linalg.norm(grav, axis=1, keepdims=True)
    av = (acc * g).sum(1); ah = np.linalg.norm(acc - av[:, None] * g, axis=1)
    gv = (gyro * g).sum(1); gh = np.linalg.norm(gyro - gv[:, None] * g, axis=1)
    return t, np.stack([av, ah, np.linalg.norm(acc, axis=1), np.linalg.norm(gyro, axis=1), gv, gh], 0).astype(np.float32)


def build_windows(real, recs, win=WIN, lag_fix=True):
    """Unlabelled windows (all walks) and labelled windows (synced walks), with device / walk ids.
    Kerb windows are centred on press + that walk's press lag (lag_fix) or press - 0.6 s (the protocol's guess)."""
    devices = sorted({w["device"] for w in real["walks"]})
    U, Ud, L, Ly, Ld, Lw, Lrule, lags = [], [], [], [], [], [], [], {}
    for w in real["walks"]:
        r = recs[w["id"]]
        a, b = walk_bounds(r, align_annotations(r)[0])
        t, X = channels(r)
        X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6)
        dev = devices.index(w["device"])
        cut = lambda c: X[:, int((c - t[0]) * FS) - win // 2: int((c - t[0]) * FS) + win // 2]
        for c in np.arange(a + 1.3, b - 1.3, 1.0):
            x = cut(c)
            if x.shape[1] == win:
                U.append(x); Ud.append(dev)
        presses = [p["t"] for p in w["labelled"]]
        if not presses:
            continue
        rule = [e["t"] for e in detect_events(r, bounds=(a, b))]
        lag, n = press_lag(w, rule) if lag_fix else (DEFAULT_LAG, 0)
        lags[w["title"]] = dict(lag_s=round(lag, 2), presses_with_jolt=n)
        fired = lambda c: any(abs(e - c) <= win / FS / 2 for e in rule)
        for p in presses:
            x = cut(p + lag)
            if x.shape[1] == win:
                L.append(x); Ly.append(1); Ld.append(dev); Lw.append(w["id"]); Lrule.append(fired(p + lag))
        rng = np.random.default_rng(len(L))
        cand = [c for c in np.arange(a + 2, b - 2, 0.5) if min(abs(c - p) for p in presses) > 4]
        for c in rng.choice(cand, min(len(cand), 3 * len(presses)), replace=False):
            x = cut(c)
            if x.shape[1] == win:
                L.append(x); Ly.append(0); Ld.append(dev); Lw.append(w["id"]); Lrule.append(fired(c))
    return dict(U=np.stack(U), Ud=np.array(Ud), L=np.stack(L), Ly=np.array(Ly), Ld=np.array(Ld), Lw=np.array(Lw),
                Lrule=np.array(Lrule), devices=devices, lags=lags)


# ----------------------------------------------------------------------------------------------- F3 / F4 / F7 models
class Trainer:
    def __init__(self, W, ft_epochs):
        import torch
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        self.W, self.ft_epochs = W, ft_epochs
        self.U, self.Ud = torch.tensor(W["U"]), torch.tensor(W["Ud"])
        self.L, self.Ly = torch.tensor(W["L"]), torch.tensor(W["Ly"])
        self.n_dev = len(W["devices"])

    def pretrain(self, seed, epochs):
        import torch
        from pathsense.ml.models import Encoder, Projector, augment, nt_xent
        torch.manual_seed(seed)
        enc, proj = Encoder(), Projector()
        opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), 1e-3)
        for _ in range(epochs):
            enc.train(); proj.train(); perm = torch.randperm(len(self.U))
            for i in range(0, len(perm), 128):
                b = self.U[perm[i:i + 128]]
                if len(b) >= 16:
                    loss = nt_xent(proj(enc(augment(b))), proj(enc(augment(b)))); opt.zero_grad(); loss.backward(); opt.step()
        return enc.state_dict()

    def finetune(self, tr, enc_state=None, dann=False, seed=0):
        import torch, torch.nn.functional as F
        from pathsense.ml.models import Encoder, DANN, augment
        torch.manual_seed(seed)
        L, Ly, U, Ud = self.L, self.Ly, self.U, self.Ud
        m = DANN(n_classes=2, n_devices=self.n_dev, enc=Encoder())
        if enc_state is not None:
            m.enc.load_state_dict(enc_state)
        o = torch.optim.Adam(m.parameters(), 1e-3)
        w = torch.tensor([1.0, float((Ly[tr] == 0).sum()) / max(int((Ly[tr] == 1).sum()), 1)])   # classes balanced
        E = self.ft_epochs
        for ep in range(E):
            m.train(); perm = tr[torch.randperm(len(tr))]
            lam = (2 / (1 + np.exp(-10 * ep / max(E - 1, 1))) - 1) if dann else 0.0
            for i in range(0, len(perm), 32):
                j = perm[i:i + 32]
                if len(j) < 4:
                    continue
                lc, _ = m(augment(L[j], jitter=0.02, max_shift=4), lam)
                loss = F.cross_entropy(lc, Ly[j], weight=w)
                if dann:                                    # device head on unlabelled windows from every phone
                    k = torch.randint(0, len(U), (64,))
                    _, ld = m(U[k], lam); loss = loss + F.cross_entropy(ld, Ud[k])
                o.zero_grad(); loss.backward(); o.step()
        return m

    def score(self, m, te):
        import torch
        from sklearn.metrics import roc_auc_score
        m.eval()
        with torch.no_grad():
            pr = torch.softmax(m(self.L[te])[0], 1)[:, 1].numpy()
        y = self.Ly[te].numpy()
        return bal_acc(y, pr > 0.5), float(roc_auc_score(y, pr)) if len(set(y.tolist())) > 1 else float("nan")


def summary(vals):
    ba, auc = zip(*vals)
    return dict(balanced_acc=round(float(np.mean(ba)), 3), sd=round(float(np.std(ba)), 3), auc=r3(np.nanmean(auc)))


def f3_f4(W, seeds, ssl_epochs, ft_epochs):
    import torch
    T = Trainer(W, ft_epochs); dn = W["devices"]; Ld, Ly = W["Ld"], W["Ly"]
    labelled = sorted(set(Ld.tolist()))
    states = {s: T.pretrain(1000 + s, ssl_epochs) for s in range(seeds)}
    res = dict(windows_unlabelled=int(len(W["U"])), windows_labelled=int(len(W["L"])), press_lags=W["lags"],
               per_phone={dn[d]: dict(kerb=int(((Ld == d) & (Ly == 1)).sum()), walking=int(((Ld == d) & (Ly == 0)).sum()))
                          for d in labelled},
               unlabelled_per_phone={dn[d]: int((W["Ud"] == d).sum()) for d in range(len(dn))})
    models = (("scratch", False, False), ("scratch+DANN", False, True), ("SSL", True, False), ("SSL+DANN", True, True))
    main = max(labelled, key=lambda d: int(((Ld == d) & (Ly == 1)).sum()))
    walk_ids = sorted(set(W["Lw"][Ld == main].tolist()))
    f3 = {}
    for frac in (0.25, 0.5, 1.0):
        for name, ssl, _ in models[::2]:
            vals = []
            for s in range(seeds):
                for held in walk_ids:
                    te = torch.tensor(np.flatnonzero(W["Lw"] == held))
                    tr_all = np.flatnonzero((Ld == main) & (W["Lw"] != held))
                    tr = torch.tensor(np.random.default_rng(s).choice(tr_all, max(int(frac * len(tr_all)), 8), replace=False))
                    vals.append(T.score(T.finetune(tr, states[s] if ssl else None, False, s), te))
            f3[f"{name} @ {int(frac * 100)}% labels"] = summary(vals)
        print(f"  F3 {int(frac * 100)}% done", flush=True)
    te_main = np.flatnonzero(Ld == main)
    f3["rule-based jolt detector (F1), same windows"] = dict(balanced_acc=round(bal_acc(Ly[te_main], W["Lrule"][te_main]), 3))
    res["F3_same_phone_leave_one_walk_out"] = dict(phone=dn[main], walks=len(walk_ids), results=f3)
    f4 = {}
    for d_tr in labelled:
        for d_te in labelled:
            if d_tr == d_te:
                continue
            tr = torch.tensor(np.flatnonzero(Ld == d_tr)); te = torch.tensor(np.flatnonzero(Ld == d_te))
            key = f"train {dn[d_tr]} -> test {dn[d_te]}"
            f4[key] = {name: summary([T.score(T.finetune(tr, states[s] if ssl else None, dann, s), te) for s in range(seeds)])
                       for name, ssl, dann in models}
            f4[key]["rule-based jolt detector (F1)"] = dict(balanced_acc=round(bal_acc(Ly[te.numpy()], W["Lrule"][te.numpy()]), 3))
            print(f"  F4 {key} done", flush=True)
    res["F4_cross_phone"] = f4
    return res


# ----------------------------------------------------------------------------------------------- F7 pre-processing
def detect(hp, t, a, b, mult, dist, fs):
    inside = (t > a + 1) & (t < b - 1)
    x = hp * inside
    steps, _ = find_peaks(x, distance=max(1, int(0.4 * fs)))
    step = float(np.median(x[steps])) if len(steps) else 1.0
    pk, _ = find_peaks(x, height=mult * step, distance=max(1, int(dist * fs)))
    return t[pk]


def f7_preprocessing(real, recs, seeds, ft_epochs):
    walks = [w for w in real["walks"] if w["labelled"]]
    FCS, MULTS, DISTS = (0.3, 1.0, 2.0, 3.0, 5.0, 8.0), (1.5, 2.0, 2.5, 3.0, 4.0), (0.3, 0.5, 1.0)
    cache = {}
    for w in walks:
        r = recs[w["id"]]; a, b = walk_bounds(r, align_annotations(r)[0]); v = vertical(r)
        rule = [e["t"] for e in detect_events(r, bounds=(a, b))]
        lag, _ = press_lag(w, rule)
        cache[w["id"]] = dict(t=r.t, a=a, b=b, fs=r.fs, lag=lag, presses=np.array([p["t"] for p in w["labelled"]]),
                              hp={fc: np.abs(highpass(v, fc, r.fs)) for fc in FCS})
    def counts(wid, fc, mult, dist):
        c = cache[wid]; det = detect(c["hp"][fc], c["t"], c["a"], c["b"], mult, dist, c["fs"])
        target = c["presses"] + c["lag"]
        tp_p = sum(np.any(np.abs(det - x) <= 1.0) for x in target)          # presses found
        tp_d = sum(np.any(np.abs(target - x) <= 1.0) for x in det)          # detections that are a marked kerb
        return tp_p, len(target), tp_d, len(det)
    def f1(cs):
        tp_p, n_p, tp_d, n_d = map(sum, zip(*cs))
        rc = tp_p / max(n_p, 1); pr = tp_d / max(n_d, 1)
        return 2 * pr * rc / max(pr + rc, 1e-9), pr, rc
    grid = [(fc, m, d) for fc in FCS for m in MULTS for d in DISTS]
    table = {g: {w["id"]: counts(w["id"], *g) for w in walks} for g in grid}
    chosen, test = [], []
    for held in walks:                                   # choose on the other walks, score on the held-out one
        best = max(grid, key=lambda g: f1([table[g][w["id"]] for w in walks if w["id"] != held["id"]])[0])
        chosen.append(best); test.append(table[best][held["id"]])
    def rep(cs):
        f, p, r = f1(cs); return dict(f1=round(f, 3), precision=round(p, 3), recall=round(r, 3))
    all_ids = [w["id"] for w in walks]
    out = dict(rule_detector=dict(
        generic_HAR_setting=dict(params="0.3 Hz high-pass (gravity separation), 2x footstep, 0.5 s spacing",
                                 **rep([table[(0.3, 2.0, 0.5)][i] for i in all_ids])),
        previous_setting=dict(params="3 Hz high-pass, 2x footstep, 0.5 s spacing", **rep([table[(3.0, 2.0, 0.5)][i] for i in all_ids])),
        re_derived_leave_one_walk_out=dict(**rep(test), chosen_per_fold=[dict(highpass_hz=g[0], threshold_x_footstep=g[1],
                                                                                min_spacing_s=g[2]) for g in chosen]),
        best_on_all_walks=dict(zip(("highpass_hz", "threshold_x_footstep", "min_spacing_s"),
                                   max(grid, key=lambda g: f1([table[g][i] for i in all_ids])[0])))))
    # window length for the learned detector (from scratch, leave-one-walk-out on the main phone)
    import torch
    wl = {}
    for win, label in ((64, "1.28 s"), (128, "2.56 s (HAR default)"), (256, "5.12 s")):
        W = build_windows(real, recs, win=win); T = Trainer(W, ft_epochs)
        main = Counter(W["Ld"][W["Ly"] == 1].tolist()).most_common(1)[0][0]
        vals = []
        for s in range(seeds):
            for held in sorted(set(W["Lw"][W["Ld"] == main].tolist())):
                te = torch.tensor(np.flatnonzero(W["Lw"] == held))
                tr = torch.tensor(np.flatnonzero((W["Ld"] == main) & (W["Lw"] != held)))
                vals.append(T.score(T.finetune(tr, None, False, s), te))
        wl[label] = summary(vals)
        print(f"  F7 window {label} done", flush=True)
    out["learned_detector_window_length"] = wl
    out["note"] = ("Derived from hand-held WALKING traces. The proposal's F7 is about wheeled motion; the same script "
                   "re-derives the settings for trolley / wheelchair recordings once they are uploaded with carry=trolley.")
    return out


# ----------------------------------------------------------------------------------------------- F5 graph propagation
def f5_graph(G, cond, real, spots, seeds, epochs):
    import torch, torch.nn.functional as F
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    edges = sorted(G.edges(data=True), key=lambda e: e[2]["idx"]); n = len(edges)
    inc = defaultdict(list)
    for u, v, d in edges:
        inc[u].append(d["idx"]); inc[v].append(d["idx"])
    A = np.eye(n)
    for es in inc.values():
        for i in es:
            for j in es:
                A[i, j] = 1
    deg = A.sum(1); A = torch.tensor(A / np.sqrt(deg[:, None] * deg[None, :]), dtype=torch.float32)
    nbrs = {i: set(j for j in np.flatnonzero(A[i].numpy()) if j != i) for i in range(n)}
    HW = ["service", "residential", "footway", "pedestrian", "steps", "track", "path", "unclassified"]
    feat = []
    for u, v, d in edges:
        mx = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2; my = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
        feat.append([float(d.get("osm_highway") == h) for h in HW] +
                    [float(d.get("osm_footway") == "sidewalk"), float(d.get("osm_footway") == "crossing"),
                     math.log1p(d["length"]) / 5, mx / 1000, my / 1000])
    feat = np.array(feat, dtype=np.float32)

    def run(y, known, K):
        idx = np.flatnonzero(known); out = defaultdict(list)
        for s in range(seeds):
            rng = np.random.default_rng(s); folds = np.array_split(rng.permutation(idx), 5)
            P = {k: np.zeros(n, int) for k in ("majority class", "features only (logistic regression)",
                                                "neighbours vote (label propagation)", "GCN (features + neighbours)")}
            for f in folds:
                vis = known.copy(); vis[f] = False
                maj = Counter(y[vis].tolist()).most_common(1)[0][0]
                P["majority class"][f] = maj
                lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(feat[vis], y[vis])
                P["features only (logistic regression)"][f] = lr.predict(feat[f])
                for i in f:
                    c = Counter(y[j] for j in nbrs[i] if vis[j])
                    P["neighbours vote (label propagation)"][i] = c.most_common(1)[0][0] if c else maj
                torch.manual_seed(s)
                W1 = torch.nn.Linear(feat.shape[1] + K + 1, 32); W2 = torch.nn.Linear(32, K)
                opt = torch.optim.Adam(list(W1.parameters()) + list(W2.parameters()), 0.01, weight_decay=5e-4)
                Xf = torch.tensor(feat); Y = torch.tensor(y)
                cw = torch.tensor([1.0 / max(int((y[vis] == k).sum()), 1) for k in range(K)], dtype=torch.float32)
                cw = cw / cw.sum() * K

                def forward(show):
                    lab = torch.zeros(n, K + 1)
                    sh = torch.tensor(np.flatnonzero(show))
                    lab[sh, Y[sh]] = 1; lab[sh, K] = 1
                    h = F.relu(W1(A @ torch.cat([Xf, lab], 1)))
                    return W2(A @ F.dropout(h, 0.2, W1.training))
                vis_idx = np.flatnonzero(vis)
                for _ in range(epochs):                     # learn to fill in hidden labels from visible neighbours
                    W1.train(); hide = rng.choice(vis_idx, max(len(vis_idx) // 4, 1), replace=False)
                    show = vis.copy(); show[hide] = False
                    loss = F.cross_entropy(forward(show)[hide], Y[hide], weight=cw)
                    opt.zero_grad(); loss.backward(); opt.step()
                W1.eval()
                with torch.no_grad():
                    P["GCN (features + neighbours)"][f] = forward(vis).argmax(1).numpy()[f]
            for k, p in P.items():
                out[k].append((float((p[idx] == y[idx]).mean()), float(f1_score(y[idx], p[idx], average="macro"))))
        return {k: dict(acc=round(float(np.mean([a for a, _ in v])), 3), macro_f1=round(float(np.mean([b for _, b in v])), 3))
                for k, v in out.items()}

    res = {}
    cls = {"smooth": 0, "paved": 1, "broken": 2, "unpaved": 2}
    y = np.zeros(n, int); known = np.zeros(n, bool)
    for i, c in cond.items():
        if c["surface_src"] in ("osm", "satellite", "team") and c["surface"] in cls:
            y[i] = cls[c["surface"]]; known[i] = True
    res["surface (smooth / paved / broken-or-unpaved)"] = dict(n=int(known.sum()), counts=dict(Counter(y[known].tolist())),
                                                              results=run(y, known, 3))
    print("  F5 surface done", flush=True)
    covered = {e for w in real["walks"] if w["labelled"] for e in w["edges"]}
    kerb = {s["edge"] for s in spots if "edge" in s}
    y = np.array([int(i in kerb) for i in range(n)]); known = np.array([i in covered for i in range(n)])
    res["kerb on segment (segments covered by synced walks)"] = dict(n=int(known.sum()), kerb_segments=int(y[known].sum()),
                                                                    results=run(y, known, 2))
    print("  F5 kerb done", flush=True)
    return res


# ----------------------------------------------------------------------------------------------- Western-centric profile
def western_comparison(root, n_pairs=300, seed=0):
    """Route many random trips with (a) PathSense's real map and (b) what an OSM-only, Western-assumption map believes:
    kerb ramps everywhere (no kerbs unless OSM tags a raised kerb), paved unless OSM tags otherwise, steps from OSM,
    nothing from walks, satellite, team reports or edits. Both routes are then scored on the REAL map."""
    from server import create_app
    from pathsense.routing import route, path_stats
    from pathsense.osm import incline_deg
    w = create_app(root=root, auto_build=False).config["WORLD"]; G = w.G
    for *_, d in G.edges(data=True):
        s = d.get("osm_surface"); surf = CND.OSM_SURFACE.get(s, "paved")
        S = CND.SURFACES[surf]
        d["west"] = dict(grad=abs(incline_deg(d.get("osm_incline")) or 0.0), rough=S["rough"], poor=S["poor"],
                         kerbs=int(d.get("osm_kerbs", 0) or 0), stairs=int(bool(d.get("steps"))), cls=0)
    nodes = list(G.nodes); rng = np.random.default_rng(seed)
    def dist(a, b):
        return math.hypot(G.nodes[a]["x"] - G.nodes[b]["x"], G.nodes[a]["y"] - G.nodes[b]["y"])
    pairs = []
    while len(pairs) < n_pairs:
        a, b = rng.choice(nodes, 2, replace=False)
        if 300 <= dist(a, b) <= 1500:
            pairs.append((a, b))
    agg = {k: defaultdict(list) for k in ("PathSense (real map)", "Western-centric (OSM + assumed ramps)", "Shortest")}
    for a, b in pairs:
        for k, attr, prof in (("PathSense (real map)", "est", "manual_wheelchair"),
                              ("Western-centric (OSM + assumed ramps)", "west", "manual_wheelchair"),
                              ("Shortest", "est", "shortest")):
            p = route(G, a, b, prof, attr)
            s = path_stats(G, p, "manual_wheelchair", "est")              # scored on what is really there
            for m, v in (("length_m", s["length"]), ("kerbs", s["kerbs"]), ("stairs", s["stairs"]),
                         ("unpaved_m", s["unpaved_m"]), ("vehicles_only_or_closed", s["closed"])):
                agg[k][m].append(v)
    out = dict(trips=n_pairs, trip_length="300-1500 m straight line, random campus junctions", profiles={})
    for k, v in agg.items():
        st = np.array(v["stairs"]); kb = np.array(v["kerbs"]); up = np.array(v["unpaved_m"]); cl = np.array(v["vehicles_only_or_closed"])
        out["profiles"][k] = dict(mean_length_m=round(float(np.mean(v["length_m"]))), trips_with_stairs_pct=round(100 * float((st > 0).mean()), 1),
                                  trips_with_a_kerb_pct=round(100 * float((kb > 0).mean()), 1), mean_kerbs=round(float(kb.mean()), 2),
                                  trips_with_unpaved_pct=round(100 * float((up > 20).mean()), 1), mean_unpaved_m=round(float(up.mean()), 1),
                                  trips_through_vehicles_only_or_closed_pct=round(100 * float((cl > 0).mean()), 1),
                                  trips_usable_by_wheelchair_pct=round(100 * float(((st == 0) & (cl == 0) & (up <= 20)).mean()), 1))
    return out


def main(a):
    t0 = time.time()
    G, real, cond, spots, recs = load_world(a.root)
    out = dict(note="Real data: 12 walks (3 phones, 3 contributors); labels from button presses and real path surfaces.")
    print("F1 inertial inference", flush=True); out["F1_inertial"] = f1_inertial(G, real, cond, recs)
    print("F2 fusion", flush=True); out["F2_fusion"] = f2_fusion(G, real, spots)
    print("F5 graph propagation", flush=True); out["F5_graph"] = f5_graph(G, cond, real, spots, a.seeds, a.gcn_epochs)
    print("Western-centric comparison", flush=True); out["Western_comparison"] = western_comparison(a.root)
    print("windows for F3/F4", flush=True); W = build_windows(real, recs)
    print(f"  {len(W['U'])} unlabelled, {len(W['L'])} labelled windows; press lags {W['lags']}", flush=True)
    out["F3_F4"] = f3_f4(W, a.seeds, a.ssl_epochs, a.ft_epochs)
    print("F7 pre-processing", flush=True); out["F7_preprocessing"] = f7_preprocessing(real, recs, max(1, a.seeds // 2), a.ft_epochs)
    out["seconds"] = round(time.time() - t0)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "real_results.json"))
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--ssl-epochs", type=int, default=20)
    ap.add_argument("--ft-epochs", type=int, default=30); ap.add_argument("--gcn-epochs", type=int, default=300)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.ssl_epochs, a.ft_epochs, a.gcn_epochs = 1, 2, 3, 30
    main(a)
