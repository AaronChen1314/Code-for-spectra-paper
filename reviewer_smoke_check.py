from pathlib import Path
import json
import numpy as np
root=Path(__file__).parent
m=json.loads((root/"parameter_manifest.json").read_text(encoding="utf-8"))
p=m["paper_alignment"]
assert (p["wbg_thickness_nm"],p["nbg_thickness_nm"])==(450.0,1200.0)
assert (p["thermal_U0_W_m-2_K-1"],p["thermal_U1_W_s_m-3_K-1"])==(14.58,4.37)
assert p["diode_temperature_exponents"]==[3.0,2.5]
for f in root.rglob("*"):
    if f.is_file() and f.suffix.lower() in {".csv",".txt",".npy",".npz",".mat",".h5",".hdf5"}: raise AssertionError(f"data file included: {f}")
try:
    from core.physics_engine import solve_single_IV_point_fast
    pmax=solve_single_IV_point_fast(18.0,19.0,298.15,tuple(m["electrical_parameters"]["WBG"]),tuple(m["electrical_parameters"]["NBG"]),0.0)
    assert np.isfinite(pmax) and pmax>0
    print(f"PASS: electrical smoke Pmax={pmax:.6g}; no spectral data included")
except ModuleNotFoundError as exc:
    if exc.name != "numba": raise
    print("PASS: manifest and data exclusion checks; optional numba solver not installed")
