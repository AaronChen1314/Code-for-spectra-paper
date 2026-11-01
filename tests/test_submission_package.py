from pathlib import Path
import ast
import json

def test_manifest_reference_values():
    root = Path(__file__).resolve().parents[1]
    p = json.loads((root / "config" / "parameter_manifest.json").read_text())["paper_alignment"]
    assert (p["wbg_thickness_nm"], p["nbg_thickness_nm"]) == (450.0, 1200.0)
    assert p["wbg_scan_eV"] == [1.70, 2.00, 0.001]

def test_package_source_syntax():
    root = Path(__file__).resolve().parents[1]
    for package in ("config", "core", "data_io", "utils", "simulation_logging", "scripts"):
        for source in (root / package).glob("*.py"):
            ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
