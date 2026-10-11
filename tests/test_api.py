"""Local product API: contract, semantics and privacy rules."""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from poker_alpha.api.server import LOCAL_ORIGINS, create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


FAST = {"equity_simulations": 150, "rollout_simulations": 150, "seed": 0,
        "use_solver": True}

ACCEPTED_STATE = {
    "num_seats": 2, "hero_seat": 0, "dealer": 0, "small_blind": 0.5,
    "big_blind": 1.0, "hero_cards": "Ah Kd", "board": "Qs 7d 2c Th 3h",
    "pot": 5.0, "actor": 0,
    "seats": [{"stack": 97.5, "committed": 2.5},
              {"stack": 97.5, "committed": 2.5}],
    "actions": [
        {"street": 0, "seat": 0, "kind": "raise", "amount": 2.5},
        {"street": 0, "seat": 1, "kind": "call", "amount": 1.5},
        {"street": 1, "seat": 1, "kind": "check"},
        {"street": 1, "seat": 0, "kind": "check"},
        {"street": 2, "seat": 1, "kind": "check"},
        {"street": 2, "seat": 0, "kind": "check"},
        {"street": 3, "seat": 1, "kind": "check"}],
}


def test_health_and_version(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["api"] == "pokeralpha.api/v1"
    assert h["solver_loaded"] is True
    v = client.get("/version").json()
    assert v["api"] == "pokeralpha.api/v1"


def test_release_info_honest(client):
    r = client.get("/release/info").json()
    assert r["iterations"] == 2_000_000
    assert len(r["strategy_sha256"]) == 64
    assert "not GTO" in r["disclaimer"]
    assert r["config_signature"].startswith("HoldemSolverConfig:")


def test_demo_spots(client):
    spots = client.get("/demo-spots").json()
    assert len(spots) >= 5
    assert all({"name", "state"} <= set(s) for s in spots)


def test_analyze_solver_semantics(client):
    r = client.post("/analyze", json={"state": ACCEPTED_STATE,
                                      "settings": FAST})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["api"] == "pokeralpha.api/v1"
    assert d["source"] == "solver" and d["solver_state"] == "ACCEPTED"
    assert d["recommended_tag"] == "HIGHEST FREQUENCY"
    assert d["ev_is_decision_basis"] is False
    assert d["actions"] and abs(sum(a["frequency"] or 0
                                    for a in d["actions"]) - 1) < 1e-6
    # EV provenance note present for solver mixes with attached estimates
    if any(a["ev_bb"] is not None for a in d["actions"]):
        assert d["provenance"]["ev_note"]
    assert d["refusal"] is None


def test_analyze_refusal(client):
    bad = {**ACCEPTED_STATE, "actor": 1}
    d = client.post("/analyze", json={"state": bad, "settings": FAST}).json()
    assert d["refusal"] is not None
    assert "turn" in d["refusal"]["message"].lower()
    assert d["actions"] == [] and d["recommended"] is None


def test_analyze_invalid_state_422(client):
    r = client.post("/analyze", json={"state": {"hello": 1}, "settings": FAST})
    assert r.status_code == 422


def test_what_if_contract(client):
    r = client.post("/what-if", json={
        "state": ACCEPTED_STATE, "variable": "hero_cards",
        "values": ["9h 9c", "Qs Qd"], "settings": FAST})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["variable"] == "hero_cards"
    assert d["variants"][0]["value"] == "base"
    assert len(d["variants"]) == 2              # Qs duplicates the board
    assert "Qs Qd" in d["rejected"]
    r2 = client.post("/what-if", json={"state": ACCEPTED_STATE,
                                       "variable": "nope", "values": ["1"],
                                       "settings": FAST})
    assert r2.status_code == 422


def test_session_review_endpoint(client):
    doc = json.loads(open("poker_alpha/data/demo_session.json").read())
    r = client.post("/session/review", json={
        "hands": doc,
        "settings": {**FAST, "equity_simulations": 150,
                     "rollout_simulations": 150}})
    assert r.status_code == 200, r.text
    rev = r.json()["review"]
    assert rev["format"] == "pokeralpha.session_review/v1"
    assert rev["summary"]["hands"] == 8
    assert "ESTIMATED DECISION REGRET" in rev["semantics"]["estimated_regret"]


def test_strategy_endpoints(client):
    m = client.get("/strategy/preflop").json()
    assert m["node"] == "BTN first action" and len(m["cells"]) == 13
    assert sum(m["stats"].values()) == 169
    one = client.get("/strategy/preflop/AKs").json()
    assert one["hand"] == "AKs"
    assert one["status"] in ("ACCEPTED", "LOW CONFIDENCE", "REJECTED",
                             "UNSUPPORTED")
    assert client.get("/strategy/preflop/ZZz").status_code == 404
    assert client.get("/strategy/preflop?node=bogus").status_code == 404


def test_no_dangerous_endpoints(client):
    """No training, filesystem or shell surface."""
    app = client.app
    paths = {r.path for r in app.routes}
    for banned in ("train", "file", "shell", "exec", "upload"):
        assert not any(banned in p for p in paths), paths
    # CORS restricted to explicit local origins
    assert all(o.startswith("http://localhost") or
               o.startswith("http://127.0.0.1") for o in LOCAL_ORIGINS)
