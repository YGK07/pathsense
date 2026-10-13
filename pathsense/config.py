"""Shared constants. Change sampling rate / window only after the pilot collection."""
FS = 50.0                      # Hz. Provisional; fix after pilot (see docs/ROADMAP.md, task P1)
G = 9.81                       # m/s^2
CLASSES = ["accessible", "caution", "blocked"]
K = len(CLASSES)

# Rule thresholds that define the *ground-truth* accessibility class of a segment.
# Slope reference: 1:12 ramp ~ 4.8 deg, 1:8 ~ 7.1 deg. Revisit with the accessibility office.
BLOCK_GRAD_DEG = 7.0
BLOCK_ROUGH = 0.75
CAUTION_GRAD_DEG = 4.0
CAUTION_ROUGH = 0.40

# Campus anchor for the demo map (VIT Vellore, approx.). Local metres -> lat/lon.
LAT0, LON0 = 12.9692, 79.1559

# Penalty weights per mobility profile: cost = d + w_g*grad_deg + w_r*roughness + w_k*kerbs
# (metres-equivalent). These are STARTING VALUES; tune with users (task R2).
PROFILES = {
    "shortest":          dict(w_g=0.0,  w_r=0.0,   w_k=0.0),
    "manual_wheelchair": dict(w_g=40.0, w_r=400.0, w_k=800.0),
    "powered_wheelchair":dict(w_g=20.0, w_r=200.0, w_k=300.0),
    "crutches":          dict(w_g=25.0, w_r=200.0, w_k=200.0),
    "low_vision":        dict(w_g=8.0,  w_r=250.0, w_k=250.0),
}
