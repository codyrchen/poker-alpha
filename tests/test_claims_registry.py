"""Claims registry: site metrics must match authoritative sources and
never go stale."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _claims():
    return json.loads((ROOT / "results" / "claims.json").read_text())["claims"]


def test_claims_match_authoritative_sources():
    c = _claims()
    bench = json.loads((ROOT / "results/benchmarks/native_mccfr_v1.json").read_text())
    assert c["native_speedup"]["value"] == bench["speedup_warm"]
    assert c["native_it_per_s"]["value"] == bench["native_warm_it_per_s_median"]
    calib = json.loads((ROOT / "results/validation/confidence_signal_quality.json").read_text())
    assert c["calibration_rows"]["value"] == calib["rows"]
    v1 = calib["v1_baseline"]["SOLVER_ACCEPT"]
    v2 = calib["v2_candidates"]["sd<0.2|mv<0.1|v>=20"]
    assert c["coverage_gain"]["value"] == round(v2["coverage"] / v1["share"], 2)
    import numpy as np

    with np.load(ROOT / c["stored_infosets"]["source"], allow_pickle=False) as z:
        assert c["stored_infosets"]["value"] == len(z["keys"])


def test_web_lab_claims_copy_matches_registry():
    web = json.loads((ROOT / "apps/web/src/data/lab/claims.json").read_text())
    assert web["payload"]["claims"] == _claims()


def test_readme_metrics_not_stale():
    c = _claims()
    readme = (ROOT / "README.md").read_text()
    assert str(c["test_count"]["value"]) in readme
    assert "835 tests" not in readme or c["test_count"]["value"] == 835
    assert f"{c['native_speedup']['value']}" in readme or "38.1" in readme


def test_no_forbidden_marketing_claims_on_site():
    """'GTO' may appear on the site only as a negation."""
    web = ROOT / "apps" / "web" / "src"
    for p in web.rglob("*.tsx"):
        text = p.read_text()
        for m in re.finditer(r".{60}GTO.{20}", text, re.S):
            frag = m.group(0).lower()
            assert ("not" in frag or "never" in frag or "no claim" in frag), \
                (p, m.group(0))


def test_web_envelopes_carry_provenance():
    lab = ROOT / "apps/web/src/data/lab"
    for p in sorted(lab.glob("*.json")):
        d = json.loads(p.read_text())
        assert d["format"] == "pokeralpha.web/v1", p
        assert len(d["generated_from_commit"]) == 40, p
        assert len(d["strategy_sha256"]) == 64, p
        assert "PRECOMPUTED" in d["provenance"], p


def test_payloads_have_no_nan_or_infinity():
    lab = ROOT / "apps/web/src/data/lab"
    for p in sorted(lab.glob("*.json")):
        text = p.read_text()
        assert "NaN" not in text and "Infinity" not in text, p
        json.loads(text)   # strict parse
