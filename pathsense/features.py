"""Functionality 1 + 7: surface/obstacle inference from raw inertial signals.

Kerbs   -> transients in high-passed vertical acceleration
Gradient-> pitch from the gravity direction (accelerometer), median over the segment
Rough   -> robust high-band energy of vertical acceleration (MAD, so kerb spikes don't inflate it)
"""
from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
from .config import FS

MIN_SEG_SECONDS = 1.5


def highpass(x, fc, fs=FS, order=4):
    return sosfiltfilt(butter(order, fc, "highpass", fs=fs, output="sos"), x)


def pitch_deg(acc):
    """Absolute slope estimate from gravity direction. acc: (N,3) with x forward."""
    p = np.arctan2(-acc[:, 0], np.sqrt(acc[:, 1] ** 2 + acc[:, 2] ** 2))
    return float(abs(np.degrees(np.median(p))))


def roughness_sigma(az, fs=FS):
    hp = highpass(az, 5.0, fs)
    return float(1.4826 * np.median(np.abs(hp - np.median(hp))))


def detect_kerbs(az, fs=FS, k=5.0, abs_thr=3.5):
    hp = highpass(az, 3.0, fs)
    sigma = 1.4826 * np.median(np.abs(hp - np.median(hp)))
    peaks, _ = find_peaks(np.abs(hp), height=max(k * sigma, abs_thr), distance=int(0.5 * fs))
    return int(len(peaks))


@dataclass
class Obs:
    trace: int
    contributor: int
    device: int
    seg: int
    kerbs: int
    grad: float
    rough_sigma: float
    duration: float
    gps_acc: float


def segment_observations(traces, fs=FS):
    """One observation per (trace, segment). Real data: replace `seg` with map-matched edge ids."""
    obs = []
    for ti, t in enumerate(traces):
        edges = np.flatnonzero(np.diff(np.r_[-1, t.seg]) != 0)
        bounds = list(edges) + [len(t.seg)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if (b - a) / fs < MIN_SEG_SECONDS:
                continue
            az = t.acc[a:b, 2]
            obs.append(Obs(ti, t.contributor, t.device, int(t.seg[a]), detect_kerbs(az, fs),
                           pitch_deg(t.acc[a:b]), roughness_sigma(az, fs), (b - a) / fs, t.gps_acc))
    return obs


def obs_matrix(obs):
    """Feature matrix for the supervised classifier: [kerbs, grad, rough_sigma]."""
    return np.array([[o.kerbs, o.grad, o.rough_sigma] for o in obs], float)
