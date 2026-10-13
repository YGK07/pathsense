"""End-to-end PathSense pipeline on SYNTHETIC data. Writes outputs/metrics.json and outputs/map.json (demo app).

    python scripts/run_pipeline.py [--traces 70] [--seed 0]
    python scripts/run_pipeline.py --osm data/osm_vit.json     # REAL campus paths (fetch with scripts/fetch_osm.py)

Only the surface data (kerbs/gradient/roughness/traces) is simulated; with --osm the geometry is real OpenStreetMap.

All numbers printed here come from simulated data and are NOT research results.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from pathsense.config import *
from pathsense.labels import label_from_attrs
from pathsense.simulate import make_campus_graph, add_surface_attributes, make_traces
from pathsense.osm import load_graph
from pathsense.features import segment_observations, obs_matrix
from pathsense.fusion import majority_vote, dawid_skene, contributor_reliability
from pathsense.graph_prop import propagate
from pathsense.routing import route, path_stats


def run(n_traces=None, seed=0, out="outputs", verbose=True, osm=None):
    rng = np.random.default_rng(seed)
    if osm:
        G = add_surface_attributes(load_graph(osm), seed=seed)
    else:
        G = make_campus_graph(seed=seed)
    n_edges = G.number_of_edges()
    if n_traces is None:                 # enough walkers to observe a realistic share of a big real graph
        n_traces = max(50, int(0.4 * n_edges)) if osm else 50
    truth = np.zeros(n_edges, int); T_rough = np.zeros(n_edges); T_grad = np.zeros(n_edges); T_k = np.zeros(n_edges)
    for u, v, d in G.edges(data=True):
        i = d["idx"]; truth[i] = d["cls"]; T_rough[i] = d["rough"]; T_grad[i] = d["grad"]; T_k[i] = d["kerbs"]
    traces = make_traces(G, n_traces=n_traces, seed=seed + 3)
    obs = segment_observations(traces)
    X = obs_matrix(obs); seg = np.array([o.seg for o in obs]); con = np.array([o.contributor for o in obs])
    y = truth[seg]
    M = {"n_edges": n_edges, "n_traces": n_traces, "n_obs": len(obs)}

    # --- F1: per-observation inference (supervised on a small labelled subset of traces) ---
    n_lab = max(int(0.2 * n_traces), 5)
    lab = np.array([o.trace < n_lab for o in obs])
    clf = RandomForestClassifier(200, min_samples_leaf=2, random_state=seed).fit(X[lab], y[lab])
    pred = clf.predict(X)
    M["obs_accuracy_unlabelled"] = float((pred[~lab] == y[~lab]).mean())
    cal = LinearRegression().fit(X[lab][:, [2]], np.array([T_rough[s] for s in seg[lab]]))
    rough_obs = np.clip(cal.predict(X[:, [2]]), 0, 1)

    # --- F2: fusion (majority vs Dawid-Skene) over segments that were observed ---
    seen = np.unique(seg); remap = {s: i for i, s in enumerate(seen)}
    items = np.array([remap[s] for s in seg])
    mv = majority_vote(items, pred, len(seen), K)
    T, CM, prior = dawid_skene(items, con, pred, len(seen), con.max() + 1, K)
    M["n_segments_observed"] = int(len(seen)); M["coverage"] = float(len(seen) / n_edges)
    M["segment_acc_single_obs"] = float((pred == y).mean())
    M["segment_acc_majority"] = float((mv.argmax(1) == truth[seen]).mean())
    M["segment_acc_dawid_skene"] = float((T.argmax(1) == truth[seen]).mean())
    rel_hat = contributor_reliability(CM, prior)
    rel_true = np.array([np.mean([t.reliability for t in traces if t.contributor == c] or [np.nan]) for c in range(con.max() + 1)])
    ok = ~np.isnan(rel_true)
    M["contributor_reliability_corr"] = float(np.corrcoef(rel_hat[ok], rel_true[ok])[0, 1])

    # fused continuous estimates, weighted by contributor reliability
    w_obs = rel_hat[con] ** 2
    est = np.zeros((n_edges, 3)); observed = np.zeros(n_edges, bool); observed[seen] = True
    kerb_obs = X[:, 0]; grad_obs = X[:, 1]
    for s in seen:
        m = seg == s; w = w_obs[m] / w_obs[m].sum()
        est[s] = [(w * rough_obs[m]).sum(), (w * grad_obs[m]).sum(), (w * kerb_obs[m]).sum()]
    est_cls = np.zeros(n_edges, int); est_cls[seen] = T.argmax(1)

    # --- F5: GCN propagation to unobserved segments ---
    pr = propagate(G, est, observed, seed=seed)
    un = ~observed
    if un.any():
        base = est[observed].mean(0)
        tgt = np.stack([T_rough, T_grad, T_k], 1)
        M["unobserved_segments"] = int(un.sum())
        M["unobs_mae_rough_gcn"] = float(np.abs(pr[un, 0] - tgt[un, 0]).mean())
        M["unobs_mae_rough_mean_baseline"] = float(np.abs(base[0] - tgt[un, 0]).mean())
        M["unobs_mae_grad_gcn"] = float(np.abs(pr[un, 1] - tgt[un, 1]).mean())
        M["unobs_mae_grad_mean_baseline"] = float(np.abs(base[1] - tgt[un, 1]).mean())
        pc = label_from_attrs(np.clip(pr[un, 1], 0, None), np.clip(pr[un, 0], 0, 1), (pr[un, 2] >= .5).astype(int))
        M["unobs_class_acc_gcn"] = float((pc == truth[un]).mean())
        M["unobs_class_acc_majority_class"] = float((np.bincount(est_cls[observed], minlength=K).argmax() == truth[un]).mean())
        est_cls[un] = pc; est[un] = pr[un]
    for u, v, d in G.edges(data=True):
        i = d["idx"]
        d["est"] = dict(grad=float(max(est[i, 1], 0)), rough=float(np.clip(est[i, 0], 0, 1)),
                        kerbs=int(round(max(est[i, 2], 0))) if observed[i] else int(est[i, 2] >= .5), cls=int(est_cls[i]))
        d["observed"] = bool(observed[i]); d["n_obs"] = int((seg == i).sum())
    M["map_class_acc_all_segments"] = float((est_cls == truth).mean())

    # --- F6: routing evaluation ---
    nodes = list(G.nodes); R = {p: dict(short_blocked=[], ps_blocked=[], orc_blocked=[], regret=[], short_regret=[]) for p in PROFILES if p != "shortest"}
    for _ in range(80):
        a, b = map(int, rng.choice(nodes, 2, replace=False))
        for p in R:
            sp = route(G, a, b, "shortest", "true"); pe = route(G, a, b, p, "est"); po = route(G, a, b, p, "true")
            st_s, st_e, st_o = (path_stats(G, q, p, "true") for q in (sp, pe, po))
            R[p]["short_blocked"].append(st_s["blocked"]); R[p]["ps_blocked"].append(st_e["blocked"]); R[p]["orc_blocked"].append(st_o["blocked"])
            R[p]["regret"].append(st_e["cost"] / st_o["cost"] - 1); R[p]["short_regret"].append(st_s["cost"] / st_o["cost"] - 1)
    M["routing"] = {p: {k: float(np.mean(v)) for k, v in d.items()} for p, d in R.items()}

    # --- export for the demo app ---
    os.makedirs(out, exist_ok=True)
    def ll(x, y_): return [LAT0 + y_ / 111320.0, LON0 + x / (111320.0 * np.cos(np.deg2rad(LAT0)))]
    def node_ll(n):                      # real graphs carry lat/lon; the fake grid is pinned on the campus anchor
        d = G.nodes[n]
        return (d["lat"], d["lon"]) if "lat" in d else tuple(ll(d["x"], d["y"]))
    nodes_j = [dict(id=n, x=G.nodes[n]["x"], y=G.nodes[n]["y"], lat=node_ll(n)[0], lon=node_ll(n)[1]) for n in G.nodes]
    edges_j = [dict(u=u, v=v, idx=d["idx"], length=d["length"], observed=d["observed"], n_obs=d["n_obs"],
                    geom=d.get("geom"), geom_from=d.get("geom_from"), name=d.get("name"), steps=bool(d.get("steps")),
                    est=d["est"], true=dict(grad=d["grad"], rough=d["rough"], kerbs=d["kerbs"], cls=d["cls"])) for u, v, d in G.edges(data=True)]
    json.dump(dict(nodes=nodes_j, edges=edges_j, profiles=PROFILES, classes=CLASSES, synthetic=True, real_geometry=bool(osm)), open(f"{out}/map.json", "w"))
    json.dump(M, open(f"{out}/metrics.json", "w"), indent=2)
    if verbose:
        print(json.dumps(M, indent=2))
    return M


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--traces", type=int, default=None); ap.add_argument("--osm", default=None, help="Overpass JSON from scripts/fetch_osm.py"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs"); a = ap.parse_args()
    run(a.traces, a.seed, a.out, osm=a.osm)
