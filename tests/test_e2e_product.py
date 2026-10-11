"""End-to-end product smoke: real API server process + the flows the
frontends depend on. No external network."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def api_proc():
    p = subprocess.Popen([sys.executable, "-m", "poker_alpha.api",
                          "--port", "0"], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True)
    port = None
    t0 = time.time()
    while time.time() - t0 < 30:
        line = p.stdout.readline()
        if line.startswith("POKERALPHA_API_PORT="):
            port = int(line.split("=")[1])
            break
    assert port, "API did not announce a port"
    base = f"http://127.0.0.1:{port}"
    t0 = time.time()
    while time.time() - t0 < 30:
        try:
            urllib.request.urlopen(base + "/health", timeout=1)
            break
        except Exception:
            time.sleep(0.2)
    yield base
    p.terminate()
    p.wait(timeout=10)


def _get(base, path):
    return json.loads(urllib.request.urlopen(base + path, timeout=60).read())


def _post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


@pytest.mark.slow
def test_e2e_api_flows(api_proc):
    base = api_proc
    # health + solver loaded
    h = _get(base, "/health")
    assert h["status"] == "ok" and h["solver_loaded"]
    # demo spot analysis — what the web analyzer and desktop overlay do
    spots = _get(base, "/demo-spots")
    r = _post(base, "/analyze", {
        "state": spots[0]["state"],
        "settings": {"equity_simulations": 200, "rollout_simulations": 200,
                     "seed": 0, "use_solver": True}})
    assert r["api"] == "pokeralpha.api/v1" and r["actions"]
    assert r["solver_state"] in ("ACCEPTED", "LOW CONFIDENCE", "REJECTED",
                                 "OFF")
    # strategy endpoint — what the web explorer uses
    m = _get(base, "/strategy/preflop")
    assert sum(m["stats"].values()) == 169
    # session review — what web Study uses
    doc = json.loads((ROOT / "poker_alpha" / "data" /
                      "demo_session.json").read_text())
    rev = _post(base, "/session/review", {
        "hands": doc,
        "settings": {"equity_simulations": 100, "rollout_simulations": 100,
                     "seed": 0, "use_solver": True}})["review"]
    assert rev["format"] == "pokeralpha.session_review/v1"
    assert rev["summary"]["hands"] == 8


@pytest.mark.slow
def test_e2e_bundled_web_data_matches_contract():
    """The web app's bundled fallback is real v1 output and stays in sync
    with the schema fields the components read."""
    data = json.loads((ROOT / "apps" / "web" / "src" / "data" /
                       "demo.json").read_text())
    assert "pre-computed" in data["generated"]
    for s in data["demo_spots"]:
        r = s["result"]
        for key in ("actions", "solver_state", "recommended_tag",
                    "ev_is_decision_basis", "provenance", "mix_meaning"):
            assert key in r, key
    assert data["release"]["iterations"] == 2_000_000
