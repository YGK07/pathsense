"""Functionality 5: propagate accessibility estimates to unobserved segments with a GCN.

Pure-numpy 2-layer GCN (manual backprop) on the LINE GRAPH of the road network (segments = nodes),
so it runs anywhere and is unit-tested. A PyTorch / GAT version lives in pathsense/ml/gnn_torch.py
(task G2). Targets are continuous per-segment attributes [roughness, gradient, kerb rate].
"""
import numpy as np, networkx as nx
import scipy.sparse as sp


def norm_adj(Gline, order):
    A = nx.to_numpy_array(Gline, nodelist=order) + np.eye(len(order))
    d = A.sum(1)
    return A / np.sqrt(d[:, None] * d[None, :])


class GCN:
    def __init__(self, din, dh, dout, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, np.sqrt(2 / din), (din, dh)); self.b1 = np.zeros(dh)
        self.W2 = r.normal(0, np.sqrt(2 / dh), (dh, dout)); self.b2 = np.zeros(dout)
        self.m = {k: 0 for k in "W1 b1 W2 b2".split()}; self.v = {k: 0 for k in self.m}; self.t = 0

    def forward(self, A, X):
        self.AX = np.asarray(A @ X)
        self.Z1 = self.AX @ self.W1 + self.b1
        self.H1 = np.maximum(self.Z1, 0)
        self.AH = np.asarray(A @ self.H1)
        return self.AH @ self.W2 + self.b2

    def step(self, A, Y, T, mask, lr=0.01):
        n = max(mask.sum(), 1)
        dY = 2 * (Y - T) * mask[:, None] / n
        g = {"W2": self.AH.T @ dY, "b2": dY.sum(0)}
        dZ1 = np.asarray(A @ (dY @ self.W2.T)) * (self.Z1 > 0)
        g["W1"] = self.AX.T @ dZ1; g["b1"] = dZ1.sum(0)
        self.t += 1
        for k, gr in g.items():                                # Adam
            self.m[k] = .9 * self.m[k] + .1 * gr; self.v[k] = .999 * self.v[k] + .001 * gr ** 2
            mh = self.m[k] / (1 - .9 ** self.t); vh = self.v[k] / (1 - .999 ** self.t)
            setattr(self, k, getattr(self, k) - lr * mh / (np.sqrt(vh) + 1e-8))


def propagate(G, est, observed, epochs=600, hide=0.25, seed=0):
    """G: road graph (edge attr 'idx'). est: (n_edges, 3) fused estimates (rows for unobserved ignored).
    observed: bool (n_edges,). Returns (n_edges,3) predictions for ALL edges (use for unobserved ones)."""
    n = G.number_of_edges()
    key = {G[u][v]["idx"]: (u, v) for u, v in G.edges}
    Lg = nx.line_graph(G)
    # sparse symmetric-normalised adjacency indexed by edge idx (real campus graphs have thousands of segments)
    r, c = [], []
    for a, b in Lg.edges:
        i, j = G[a[0]][a[1]]["idx"], G[b[0]][b[1]]["idx"]
        r += [i, j]; c += [j, i]
    A_idx = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
    A_idx.data[:] = 1.0
    A_idx = A_idx + sp.eye(n, format="csr"); d = np.asarray(A_idx.sum(1)).ravel()
    A = sp.diags(d ** -0.5) @ A_idx @ sp.diags(d ** -0.5)
    mid = np.array([[(G.nodes[key[i][0]]["x"] + G.nodes[key[i][1]]["x"]) / 2,
                     (G.nodes[key[i][0]]["y"] + G.nodes[key[i][1]]["y"]) / 2] for i in range(n)])
    mid = (mid - mid.mean(0)) / mid.std(0)
    mu, sd = est[observed].mean(0), est[observed].std(0) + 1e-6
    Tn = (est - mu) / sd
    rng = np.random.default_rng(seed)
    net = GCN(3 + 1 + 2, 32, 3, seed)
    obs_idx = np.flatnonzero(observed)

    def build(vis):
        X = np.zeros((n, 6)); X[vis, :3] = Tn[vis]; X[vis, 3] = 1; X[:, 4:] = mid
        return X
    for _ in range(epochs):
        hid = rng.choice(obs_idx, max(int(hide * len(obs_idx)), 1), replace=False)
        vis = observed.copy(); vis[hid] = False
        mask = np.zeros(n); mask[hid] = 1
        Y = net.forward(A, build(vis))
        net.step(A, Y, np.where(observed[:, None], Tn, 0), mask)
    return net.forward(A, build(observed)) * sd + mu
