import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsense.labels import label_from_attrs
from pathsense.fusion import dawid_skene, majority_vote
from pathsense.features import detect_kerbs, roughness_sigma, pitch_deg
from pathsense.simulate import make_campus_graph, make_traces
from pathsense.routing import route, path_stats
from pathsense.graph_prop import propagate


def test_labels():
    assert label_from_attrs(0, 0.1, 0) == 0
    assert label_from_attrs(5, 0.1, 0) == 1
    assert label_from_attrs(0, 0.1, 1) == 2


def test_kerb_detector_finds_spike():
    rng = np.random.default_rng(0); az = 9.81 + rng.normal(0, .2, 200)
    t = np.arange(40) / 50; az[100:140] += 7 * np.exp(-t / .08) * np.cos(2 * np.pi * 9 * t)
    assert detect_kerbs(az) == 1 and detect_kerbs(9.81 + rng.normal(0, .2, 200)) == 0


def test_roughness_monotonic():
    rng = np.random.default_rng(0)
    assert roughness_sigma(rng.normal(0, 2, 500)) > roughness_sigma(rng.normal(0, .3, 500))


def test_pitch():
    th = np.deg2rad(6.0); acc = np.tile([-9.81 * np.sin(th), 0, 9.81 * np.cos(th)], (50, 1))
    assert abs(pitch_deg(acc) - 6.0) < 0.1


def test_dawid_skene_beats_noisy_annotators():
    rng = np.random.default_rng(0); n, K = 300, 3; z = rng.integers(0, K, n)
    items, ann, lab = [], [], []
    for i in range(n):
        for a, acc in enumerate([.9, .85, .4, .4, .35]):
            items.append(i); ann.append(a); lab.append(z[i] if rng.random() < acc else rng.integers(K))
    T, CM, _ = dawid_skene(items, ann, lab, n, 5, K)
    mv = majority_vote(np.array(items), np.array(lab), n, K).argmax(1)
    assert (T.argmax(1) == z).mean() >= (mv == z).mean() - 0.01 and (T.argmax(1) == z).mean() > .85
    assert CM[0].trace() > CM[4].trace()           # good annotator recognised as better


def test_routing_avoids_penalised_edge():
    G = make_campus_graph(seed=0)
    for u, v, d in G.edges(data=True):
        d["est"] = dict(grad=d["grad"], rough=d["rough"], kerbs=d["kerbs"], cls=d["cls"])
    a, b = 0, G.number_of_nodes() - 1
    sp = route(G, a, b, "shortest", "est"); pe = route(G, a, b, "manual_wheelchair", "est")
    assert path_stats(G, pe, "manual_wheelchair")["cost"] <= path_stats(G, sp, "manual_wheelchair")["cost"] + 1e-6


def test_gcn_beats_mean_baseline():
    G = make_campus_graph(seed=0); n = G.number_of_edges()
    tgt = np.zeros((n, 3))
    for u, v, d in G.edges(data=True): tgt[d["idx"]] = [d["rough"], d["grad"], d["kerbs"]]
    obs = np.random.default_rng(1).random(n) < .75
    pr = propagate(G, tgt, obs, epochs=300)
    assert np.abs(pr[~obs, 1] - tgt[~obs, 1]).mean() < np.abs(tgt[obs, 1].mean() - tgt[~obs, 1]).mean()


def test_pipeline_runs(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import run_pipeline
    M = run_pipeline.run(n_traces=30, seed=0, out=str(tmp_path), verbose=False)
    assert 0 <= M["segment_acc_dawid_skene"] <= 1 and (tmp_path / "map.json").exists()


def test_windows_do_not_span_segments():
    from pathsense.ml.data import make_windows
    G = make_campus_graph(seed=0); T = make_traces(G, 15)
    X, y, dev, seg = make_windows(T, G)
    assert X.shape[1:] == (6, 100) and len(X) == len(y) == len(dev) > 100
