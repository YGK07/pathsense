import numpy as np
from .config import BLOCK_GRAD_DEG, BLOCK_ROUGH, CAUTION_GRAD_DEG, CAUTION_ROUGH

def label_from_attrs(grad_deg, rough, kerbs):
    """Profile-agnostic accessibility class from physical attributes (scalars or arrays)."""
    grad_deg, rough, kerbs = map(np.asarray, (grad_deg, rough, kerbs))
    blocked = (kerbs >= 1) | (grad_deg >= BLOCK_GRAD_DEG) | (rough >= BLOCK_ROUGH)
    caution = (grad_deg >= CAUTION_GRAD_DEG) | (rough >= CAUTION_ROUGH)
    return np.where(blocked, 2, np.where(caution, 1, 0)).astype(int)
