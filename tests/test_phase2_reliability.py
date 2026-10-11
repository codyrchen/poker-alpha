"""Phase II Program 20: reliability under missing components and bad inputs."""

import builtins
import json
import sys
import types

import pytest

from poker_alpha import native as native_pkg
from poker_alpha.research import validate as V
from poker_alpha.solver_config import RELEASE_CONFIG


def test_missing_native_extension_falls_back_to_python(monkeypatch):
    from poker_alpha.native import backend
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "poker_alpha_native":
            raise ImportError("simulated missing extension")
        return real_import(name, *a, **k)
    monkeypatch.delitem(sys.modules, "poker_alpha_native", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert native_pkg.native_module() is None and not native_pkg.native_available()
    solver, name = backend.build_solver(RELEASE_CONFIG, seed=0, backend="auto")
    assert name == "python" and solver.iterations == 0
    with pytest.raises(backend.NativeBackendError):
        backend.build_solver(RELEASE_CONFIG, seed=0, backend="native")


def test_incompatible_native_schema_is_rejected(monkeypatch):
    fake = types.ModuleType("poker_alpha_native")
    fake.BACKEND_SCHEMA = 999
    monkeypatch.setitem(sys.modules, "poker_alpha_native", fake)
    assert native_pkg.native_module() is None


def test_partial_and_truncated_imports_raise_clear_errors():
    from poker_alpha.history.importers import ImportFormatError, simple_lines
    good = ("# pokeralpha-lines/v1\nhand h1 seats=2 dealer=0 sb=0.5 bb=1 hero=0 stacks=100,100\n"
            "deal 0 Ah Kd\nraise 0 2.5\ncall 1\nflop Qs 7d 2c\ncheck 1\nbet 0 3.5\nfold 1\n")
    assert len(simple_lines.parse(good)) == 1
    for bad in (good.replace("flop Qs 7d 2c", "flop Qs 7d"),       # truncated board
                good.replace("deal 0 Ah Kd", "deal 0 Ah"),         # truncated cards
                good.replace("hand h1", "hnd h1"),                 # corrupted header line
                "# pokeralpha-lines/v1\ndeal 0 Ah Kd\n"):          # events before any hand
        with pytest.raises((ImportFormatError, ValueError)):
            simple_lines.parse(bad)


def test_research_artifact_validator(tmp_path):
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"format": "pokeralpha.phase2.x/v1", "commit": "a" * 40, "runs": [1]}))
    assert V.validate_file(ok) == []
    nan = tmp_path / "nan.json"
    nan.write_text('{"format": "pokeralpha.phase2.x/v1", "v": NaN}')
    assert "strict JSON" in V.validate_file(nan)[0]
    nofmt = tmp_path / "nofmt.json"
    nofmt.write_text(json.dumps({"runs": []}))
    probs = V.validate_file(nofmt)
    assert any("format" in p for p in probs) and any("empty runs" in p for p in probs)
    # every committed Phase II artifact validates
    assert V.validate() == []
