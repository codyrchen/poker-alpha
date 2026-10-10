"""Phase 53: calibration file versioning, migration and integrity."""

import json
from pathlib import Path

import pytest

from poker_alpha.observer.calibration import (CALIBRATION_FORMAT, CALIBRATION_V1,
                                              TableCalibration)
from poker_alpha.observer.errors import CalibrationError
from poker_alpha.observer.pokernow import default_layout, pokernow_hu_layout

FIX = Path(__file__).parent / "fixtures"


def test_v2_round_trip_and_provenance(tmp_path):
    cal = pokernow_hu_layout()
    p = tmp_path / "c.json"
    cal.save(p)
    d = json.loads(p.read_text())
    assert d["format"] == CALIBRATION_FORMAT and d["schema_version"] == 2
    assert d["created"] and d["updated"] and "tuning frame" in d["source"]
    assert d["checksum"] == cal.geometry_checksum() and "fractions of the table box" in \
        d["region_semantics"]
    back = TableCalibration.load(p)
    assert back.geometry_payload() == cal.geometry_payload() and back.migrated_from is None


def test_checksum_ignores_provenance_but_tracks_geometry(tmp_path):
    cal = default_layout(6, 2)
    c0 = cal.geometry_checksum()
    cal.save(tmp_path / "a.json")
    cal.save(tmp_path / "b.json")                   # new timestamps, same geometry
    assert cal.geometry_checksum() == c0
    cal.felt_tolerance += 1
    assert cal.geometry_checksum() != c0


def test_v1_files_migrate_with_documented_defaults():
    d = json.loads((FIX / "table_calibration.json").read_text())
    assert d["format"] == CALIBRATION_V1
    cal = TableCalibration.from_dict(d)
    assert cal.migrated_from == "v1"
    assert (cal.table_detector, cal.client, cal.pot_includes_bets) == \
        ("felt_color", "generic", True)
    assert {k: r.to_list() for k, r in cal.regions.items()} == d["regions"]   # geometry kept
    re = TableCalibration.from_dict(cal.to_dict())
    assert re.migrated_from == "v1" and re.geometry_payload() == cal.geometry_payload()


def test_tampered_geometry_is_refused(tmp_path):
    cal = pokernow_hu_layout()
    p = tmp_path / "c.json"
    cal.save(p)
    d = json.loads(p.read_text())
    d["regions"]["pot"][0] += 0.05
    p.write_text(json.dumps(d))
    with pytest.raises(CalibrationError, match="checksum mismatch"):
        TableCalibration.load(p)
    assert TableCalibration.load(p, verify=False).regions["pot"].x == pytest.approx(
        cal.regions["pot"].x + 0.05)


def test_unknown_fields_newer_versions_and_garbage_are_refused():
    d = pokernow_hu_layout().to_dict()
    with pytest.raises(CalibrationError, match="unknown fields"):
        TableCalibration.from_dict(dict(d, magic_offset=3))
    with pytest.raises(CalibrationError, match="newer than"):
        TableCalibration.from_dict(dict(d, format="pokeralpha.calibration/v3"))
    with pytest.raises(CalibrationError, match="unsupported"):
        TableCalibration.from_dict(dict(d, format="something-else"))
    bad = dict(d)
    del bad["checksum"]
    with pytest.raises(CalibrationError, match="without checksum"):
        TableCalibration.from_dict(bad)
    missing = dict(d)
    del missing["felt_color"]
    with pytest.raises(CalibrationError, match="missing field"):
        TableCalibration.from_dict(missing, verify=False)


def test_committed_pokernow_calibration_is_v2_and_verifies():
    cal = TableCalibration.load(FIX / "pokernow" / "calibration.json")
    assert cal.geometry_payload() == pokernow_hu_layout().geometry_payload()
