"""Phase 24: end-to-end demos are runnable and deterministic."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run([sys.executable, "-m", *args], capture_output=True,
                          text=True, check=True, cwd=ROOT).stdout


def test_holdem_demo_end_to_end_and_deterministic():
    a = run("poker_alpha.holdem_demo", "--rollouts", "300", "--equity-sims", "500")
    b = run("poker_alpha.holdem_demo", "--rollouts", "300", "--equity-sims", "500")
    assert a == b
    for step in ("Table (hero's view", "Opponent ranges", "blocks", "Equity, pot odds",
                 "Candidate actions", "DecisionReport", "Recommendation:"):
        assert step in a
    assert "GTO" not in a and "validation: ok" in a


@pytest.mark.vision
def test_observer_demo_on_fixture():
    pytest.importorskip("PIL")
    out = run("poker_alpha.observer.demo", "tests/fixtures/table.png",
              "--rollouts", "200")
    assert "Recognized fields" in out and "hero_card_0" in out
    assert "As Ks" in out and "board [Qs Js 4h]" in out
    assert "=== Validation ===\n  ok" in out
    assert "Recommendation:" in out and "not validated" in out
    quiet = run("poker_alpha.observer.demo", "tests/fixtures/table.png", "--no-decide")
    assert "Recommendation:" not in quiet
