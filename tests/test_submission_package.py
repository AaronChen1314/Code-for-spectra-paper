from pathlib import Path
import json

def test_manifest_matches_manuscript():
    root = Path(__file__).resolve().parents[1]
    p = json.loads((root / "config" / "parameter_manifest.json").read_text())["paper_alignment"]
    assert (p["wbg_thickness_nm"], p["nbg_thickness_nm"]) == (450.0, 1200.0)
    assert p["wbg_scan_eV"] == [1.70, 2.00, 0.001]

def test_no_spectral_files_are_distributed():
    root = Path(__file__).resolve().parents[1]
    forbidden = {".csv", ".npy", ".npz", ".mat", ".h5", ".hdf5"}
    assert not [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in forbidden]
