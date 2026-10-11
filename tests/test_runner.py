"""Phase II: experiment orchestration (hash, cache, timeout, failure, registry, claim gate)."""

import json

import pytest

from poker_alpha.research import runner
from poker_alpha.research._runner_examples import boom, sleepy, square


def test_content_hash_is_stable_and_sensitive():
    a = runner.content_hash("k", {"x": 1, "y": [1, 2]}, seed=0)
    assert a == runner.content_hash("k", {"y": [1, 2], "x": 1}, seed=0)
    assert a != runner.content_hash("k", {"x": 2, "y": [1, 2]}, seed=0)
    assert a != runner.content_hash("k", {"x": 1, "y": [1, 2]}, seed=1)
    assert a != runner.content_hash("k2", {"x": 1, "y": [1, 2]}, seed=0)
    assert a != runner.content_hash("k", {"x": 1, "y": [1, 2]}, seed=0, schema=99)


def test_input_artifact_hash_changes_the_task_hash(tmp_path, monkeypatch):
    f = tmp_path / "in.bin"
    f.write_bytes(b"one")
    h1 = runner.content_hash("k", {}, [str(f)])
    f.write_bytes(b"two")
    assert runner.content_hash("k", {}, [str(f)]) != h1


def test_cache_timeout_and_failure_records(tmp_path):
    t = runner.Task("square", {"x": 3}, seed=1)
    r1 = runner.run_task(square, t, cache_dir=tmp_path)
    assert r1.status == "ok" and r1.result == {"value": 16}
    r2 = runner.run_task(square, t, cache_dir=tmp_path)
    assert r2.status == "cached" and r2.result == {"value": 16}
    r3 = runner.run_task(boom, runner.Task("boom", {}, seed=2), cache_dir=tmp_path)
    assert r3.status == "error" and "deliberate failure" in r3.error
    r4 = runner.run_task(sleepy, runner.Task("sleepy", {"seconds": 30}, timeout=1.0), cache_dir=tmp_path)
    assert r4.status == "timeout" and r4.seconds < 15


def test_grid_continues_after_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CACHE_DIR", tmp_path)
    res = runner.run_grid(square, "square", {"x": 1}, seeds=[0, 1, 2], workers=3, use_cache=False)
    assert [r.result["value"] for r in res] == [1, 4, 9]


def test_registry_record_verify_and_claim_gate(tmp_path, monkeypatch):
    idx = tmp_path / "index.json"
    out = tmp_path / "out.json"
    out.write_text("{}")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    runner.record("disc", status="done", outputs=["out.json"], classification="POSITIVE", mode="discovery", path=idx)
    assert runner.verify(idx) == []
    out.write_text('{"changed": 1}')
    assert any("changed since recording" in p for p in runner.verify(idx))
    claims = tmp_path / "claims.json"
    claims.write_text(json.dumps({"format": "pokeralpha.claims/v1", "claims": {}}))
    with pytest.raises(ValueError, match="only confirmed"):
        runner.promote_claim("c", 1, "x", "s", "q", "disc", index_path=idx, claims_path=claims)
    runner.record("conf", status="done", outputs=[], classification="POSITIVE", mode="confirmation", path=idx)
    runner.promote_claim("c", 1, "x", "s", "q", "conf", index_path=idx, claims_path=claims)
    assert json.loads(claims.read_text())["claims"]["c"]["experiment"] == "conf"
    with pytest.raises(ValueError):
        runner.record("bad", status="done", outputs=[], classification="AMAZING", path=idx)
