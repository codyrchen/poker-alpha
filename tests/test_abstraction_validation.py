"""Phase 25: encoder-independent compression and perfect-recall audit."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from poker_alpha.abstraction import (HoldemBucketEncoder, RawHoldemEncoder,
                                     ToyHoldemEncoder)
from poker_alpha.games import HoldemGame
from poker_alpha.validation.abstraction_audit import (audit_perfect_recall,
                                                      generate_corpus,
                                                      generate_line_corpus,
                                                      measure_compression,
                                                      prior_decisions)
from poker_alpha.validation.canonical_spots import canonical_spots, spot_policies

ROOT = Path(__file__).resolve().parents[1]
FAST_BUCKET = HoldemBucketEncoder(equity_samples=40)


@pytest.fixture(scope="module")
def game():
    return HoldemGame()


@pytest.fixture(scope="module")
def corpus(game):
    return generate_corpus(game, 250, seed=3)


def test_corpus_is_frozen_and_encoder_independent(game, corpus):
    again = generate_corpus(game, 250, seed=3)
    assert [(s.holes, s.board, s.streets) for s in again] == \
        [(s.holes, s.board, s.streets) for s in corpus]
    raws = {(s.holes, s.board, s.streets) for s in corpus}
    assert len(raws) == len(corpus)                       # deduplicated
    streets = {len(s.streets) for s in corpus}
    assert streets == {1, 2, 3, 4}                         # every street present
    # Changing the encoder must not change the corpus.
    other = generate_corpus(HoldemGame(encoder=ToyHoldemEncoder()), 250, seed=3)
    assert [(s.holes, s.board, s.streets) for s in other] == \
        [(s.holes, s.board, s.streets) for s in corpus]


def test_abstractions_never_exceed_raw_keys(game, corpus):
    rep = measure_compression(game, corpus, {"toy": ToyHoldemEncoder(),
                                             "bucket": FAST_BUCKET})
    assert rep.states >= rep.raw_keys
    for name in ("toy", "bucket"):
        assert rep.keys[name] <= rep.raw_keys
        assert rep.invariant_violations[name] == 0
        for street, row in rep.by_street.items():
            assert row[name] <= row["raw"] <= row["states"], (name, street)
    assert rep.keys["toy"] < rep.keys["bucket"]
    assert set(rep.by_street) == {"preflop", "flop", "turn", "river"}
    assert sum(r["states"] for r in rep.by_position.values()) == rep.states


def test_preflop_fixed_line_collapses_to_169_classes(game):
    lc = generate_line_corpus(game, ("",), 800, seed=1)
    rep = measure_compression(game, lc, {"toy": ToyHoldemEncoder(),
                                         "bucket": FAST_BUCKET})
    assert rep.keys["bucket"] <= 169 and rep.keys["toy"] <= 6
    assert rep.raw_keys > rep.keys["bucket"]


def test_perfect_recall_audit(game, corpus):
    raw = audit_perfect_recall(game, corpus, RawHoldemEncoder(), "raw")
    bucket = audit_perfect_recall(game, corpus, FAST_BUCKET, "bucket")
    toy = audit_perfect_recall(game, corpus, ToyHoldemEncoder(), "toy")
    assert raw.perfect_recall and bucket.perfect_recall
    assert bucket.colliding_keys > 0                      # the audit had work to do
    assert not toy.perfect_recall
    assert toy.violation_kinds["earlier_abstraction"] == toy.violations
    text = toy.examples[0].format()
    for line in ("abstract key:", "member count:", "own action history consistent",
                 "public observation history consistent",
                 "earlier private abstraction consistent", "legal action set consistent"):
        assert line in text


class _ForgetsHistory:
    """Deliberately broken encoder: drops the action history."""

    def encode(self, game, state):
        from poker_alpha.abstraction import preflop_class
        p = game.current_player(state)
        return f"{p}|{preflop_class(state.holes[p])}|{len(state.streets)}"

    def signature(self):
        return "ForgetsHistory:v1"


def test_audit_detects_forgotten_actions(game, corpus):
    a = audit_perfect_recall(game, corpus, _ForgetsHistory(), "broken")
    assert not a.perfect_recall
    assert a.violation_kinds["public_history"] > 0
    assert a.violation_kinds["own_actions"] > 0


def test_prior_decisions_reconstruct_own_history(game):
    lc = generate_line_corpus(game, ("b100c", "cb50"), 5, seed=2)
    s = lc[0]
    assert game.current_player(s) == 1
    prior = prior_decisions(game, s)
    assert [t for _, t in prior] == ["c", "c"]          # BB's call, BB's check
    assert [len(p.board) for p, _ in prior] == [0, 3]


def test_canonical_spots_are_well_formed():
    game = HoldemGame(bet_fractions={"b33": 0.33, "b75": 0.75, "b150": 1.5},
                      encoder=FAST_BUCKET)
    pols = spot_policies(game, {}, canonical_spots())
    assert len(pols) == 14
    for p in pols:
        assert not p.visited and p.l1_from_uniform == pytest.approx(0.0)
        assert sum(p.probs) == pytest.approx(1.0)
    keys = {p.key for p in pols}
    assert "0|AA|" in keys and any(k.endswith("b75c/cb75") for k in keys)


def test_mccfr_probe_script(tmp_path):
    out = tmp_path / "run.jsonl"
    subprocess.run([sys.executable, "experiments/holdem_mccfr_validation.py",
                    "--seed", "0", "--milestones", "2,4", "--ckpt-dir", str(tmp_path),
                    "--out", str(out)], check=True, cwd=ROOT, capture_output=True)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["iterations"] for r in rows] == [2, 4]
    assert rows[1]["new_infosets"] == rows[1]["infosets"] - rows[0]["infosets"]
    vf = rows[1]["visit_fraction"]
    assert sum(vf[k] for k in ("0", "1", "2-5", "6-20", ">20")) == pytest.approx(1.0)
    assert rows[1]["top_n_prev_mean_l1"] is not None
