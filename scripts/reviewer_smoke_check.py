"""Offline reviewer check: parameters, solver import, and data exclusion."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "config" / "parameter_manifest.json").read_text(encoding="utf-8"))
p = manifest["paper_alignment"]
assert (p["wbg_thickness_nm"], p["nbg_thickness_nm"]) == (450.0, 1200.0)
assert (p["thermal_U0_W_m-2_K-1"], p["thermal_U1_W_s_m-3_K-1"]) == (14.58, 4.37)
assert p["diode_temperature_exponents"] == [3.0, 2.5]
for path in ROOT.rglob("*"):
    if path.is_file() and path.suffix.lower() in {".csv", ".npy", ".npz", ".mat", ".h5", ".hdf5"}:
        raise AssertionError(f"spectral/data file included: {path}")
try:
    import sys
    sys.path.insert(0, str(ROOT))
    from core.physics_engine import solve_single_IV_point_fast
    pmax = solve_single_IV_point_fast(18.0, 19.0, 298.15,
        tuple(manifest["electrical_parameters"]["WBG"]),
        tuple(manifest["electrical_parameters"]["NBG"]), 0.0)
    assert np.isfinite(pmax) and pmax > 0
    print(f"PASS: electrical smoke Pmax={pmax:.6g}; no spectral data included")
except ModuleNotFoundError as exc:
    if exc.name != "numba":
        raise
    print("PASS: manifest and data exclusion checks; optional numba solver not installed")
