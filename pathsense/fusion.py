"""Functionality 2: multi-trace fusion as a Dawid-Skene multi-annotator model (EM).

Each contributor is an annotator with an unknown confusion matrix; each segment has an unknown true class.
"""
import numpy as np


def majority_vote(items, labels, n_items, K):
    cnt = np.zeros((n_items, K))
    np.add.at(cnt, (items, labels), 1)
    return cnt


def dawid_skene(items, annotators, labels, n_items, n_annot, K, iters=60, smooth=0.5, tol=1e-6):
    """items/annotators/labels: int arrays (one entry per observation).
    Returns posterior (n_items,K), confusion matrices (n_annot,K,K), class prior (K,)."""
    items, annotators, labels = map(np.asarray, (items, annotators, labels))
    T = majority_vote(items, labels, n_items, K) + 1e-3
    T /= T.sum(1, keepdims=True)
    prev = None
    for _ in range(iters):
        prior = T.mean(0) + 1e-9
        CM = np.full((n_annot, K, K), smooth) + 4 * smooth * np.eye(K)[None]   # mild prior: annotators are better than chance
        np.add.at(CM, (annotators[:, None], np.arange(K)[None, :], labels[:, None]), T[items])
        CM /= CM.sum(2, keepdims=True)
        logT = np.tile(np.log(prior), (n_items, 1))
        np.add.at(logT, items, np.log(CM[annotators, :, labels]))
        logT -= logT.max(1, keepdims=True)
        T = np.exp(logT); T /= T.sum(1, keepdims=True)
        if prev is not None and np.abs(T - prev).max() < tol:
            break
        prev = T
    return T, CM, prior


def contributor_reliability(CM, prior):
    """Scalar reliability per annotator = prior-weighted diagonal of its confusion matrix."""
    return (CM[:, np.arange(CM.shape[1]), np.arange(CM.shape[1])] * prior[None]).sum(1)
