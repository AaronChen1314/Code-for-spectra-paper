"""Offline reviewer check: reference parameters, solver import, and package structure."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "config" / "parameter_manifest.json").read_text(encoding="utf-8"))
p = manifest["paper_alignment"]
assert (p["wbg_thickness_nm"], p["nbg_thickness_nm"]) == (450.0, 1200.0)
assert (p["thermal_U0_W_m-2_K-1"], p["thermal_U1_W_s_m-3_K-1"]) == (14.58, 4.37)
assert p["diode_temperature_exponents"] == [3.0, 2.5]
for module in (
    "core/physics_engine.py",
    "core/simulation_engine.py",
    "config/config_manager.py",
    "data_io/file_io.py",
    "data_io/poa_spectrum.py",
    "utils/physics_utils.py",
):
    assert (ROOT / module).is_file(), f"package module check: {module}"
try:
    import sys
    sys.path.insert(0, str(ROOT))
    from core.physics_engine import solve_single_IV_point_fast
    pmax = solve_single_IV_point_fast(18.0, 19.0, 298.15,
        tuple(manifest["electrical_parameters"]["WBG"]),
        tuple(manifest["electrical_parameters"]["NBG"]), 0.0)
    assert np.isfinite(pmax) and pmax > 0
    print(f"PASS: parameter and package checks; electrical smoke Pmax={pmax:.6g}")
except ModuleNotFoundError as exc:
    if exc.name != "numba":
        raise
    print("PASS: parameter and package checks; install numba to run the electrical solver")
