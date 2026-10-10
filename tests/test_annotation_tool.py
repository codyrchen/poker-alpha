"""Phases 43, 45, 46: session annotation, dataset splits and frame metrics.

All images are SYNTHETIC (rendered tables); nothing here measures real
PokerNow accuracy."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")

from observer_helpers import frames_of, record, table  # noqa: E402

from poker_alpha.observer.annotation_tool import (SessionAnnotator,  # noqa: E402
                                                  build_annotation, parse_card,
                                                  parse_cards, seat_fields_to_status,
                                                  seat_status_to_fields)
from poker_alpha.observer.annotations import (annotation_from_dict,  # noqa: E402
                                              confidence_buckets, score, score_by_role,
                                              validate_annotation)
from poker_alpha.observer.pokernow import (FieldReading, FrameObservation,  # noqa: E402
                                           )

pytestmark = pytest.mark.vision
ROOT = Path(__file__).resolve().parents[1]


def test_card_parsing():
    assert parse_card("js") == "Js" and parse_card("10S") == "Ts" and parse_card("T♠") == "Ts"
    assert parse_card("") is None
    assert parse_cards("") is None and parse_cards("-") == []
    assert parse_cards("Qs, jh 4c") == ["Qs", "Jh", "4c"]
    for bad in ("Xx", "Q", "Qsx"):
        with pytest.raises((ValueError, KeyError)):
            parse_card(bad)


@pytest.mark.parametrize("status", ["unknown", "empty", "in hand", "folded", "all-in"])
def test_seat_status_round_trip(status):
    assert seat_fields_to_status(seat_status_to_fields(status)) == status


BASE = {"format": "pokeralpha.screenshot_annotation/v1", "image": "frames/000001.png",
        "num_seats": 2, "hero_seat": 0}


def test_partial_annotation_scores_only_known_fields():
    d = build_annotation(BASE, status="partial", hero_cards=("As", "kd"))
    assert d["annotated"] == ["hero_cards"] and validate_annotation(d) == []
    ann = annotation_from_dict(d, "x", Path("x.png"))
    assert ann.board is None and ann.seats[1].occupied is None
    obs = FrameObservation(None, (0, 0, 1, 1))
    obs.fields["hero_card_0"] = FieldReading("As", 0.9, "hero_card_0")
    obs.fields["hero_card_1"] = FieldReading("Kh", 0.4, "hero_card_1")
    m = score([(obs, ann)])
    assert m["hero_card"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert m["card_rank"]["accuracy"] == 1.0 and m["card_suit"]["accuracy"] == 0.5
    for k in ("board_card", "stack", "pot", "occupied", "dealer", "street"):
        assert m[k]["total"] == 0, k
    assert m["full_state"]["accuracy"] == 0.0


def test_build_annotation_seats_and_preflop_street():
    seats = [{"status": "in hand", "stack": "99", "bet": "0.5"},
             {"status": "all-in", "stack": "", "bet": "20"}]
    d = build_annotation(BASE, status="complete", role="validation", street="preflop",
                         dealer="1", actor="none", pot="1.5", seats=seats, notes=" hi ")
    assert d["board"] == [] and "board" in d["annotated"]
    assert d["seats"][1] == {"seat": 1, "occupied": True, "active": True, "all_in": True,
                             "bet": 20.0, "stack": 0.0}
    assert d["actor"] is None and "actor" in d["annotated"] and d["notes"] == "hi"
    assert validate_annotation(d) == []
    with pytest.raises(ValueError):
        build_annotation(BASE, status="complete", hero_cards=("As", ""))
    with pytest.raises(ValueError):
        build_annotation(BASE, status="bogus")


def test_no_validation_frames_means_blocked_not_zero_or_hundred():
    ann = annotation_from_dict(dict(BASE, role="tuning", hero_cards=["As", "Kd"],
                                    seats=[{"seat": 0, "stack": 1.0}]), "t", Path("t.png"))
    obs = FrameObservation(None, (0, 0, 1, 1))
    res = score_by_role([(obs, ann)])
    assert res["real_validation"]["status"] == "REAL VALIDATION: BLOCKED"
    assert "metrics" not in res["real_validation"]
    assert "TUNING-FIT ONLY" in res["tuning_fit"]["label"]
    unassigned = annotation_from_dict(dict(BASE, hero_cards=["As", "Kd"],
                                           seats=[{"seat": 0, "stack": 1.0}]), "u",
                                      Path("u.png"))
    res = score_by_role([(obs, unassigned)])
    assert res["real_validation"]["status"] == "REAL VALIDATION: BLOCKED"
    assert res["unassigned"]["frames"] == 1
    val = annotation_from_dict(dict(BASE, role="held_out", hero_cards=["As", "Kd"],
                                    seats=[{"seat": 0, "stack": 1.0}]), "v", Path("v.png"))
    res = score_by_role([(obs, ann), (obs, val)])
    assert res["real_validation"]["status"] == "MEASURED"
    assert res["real_validation"]["frames"] == 1                # tuning frame excluded


def test_confidence_buckets_cover_unit_interval():
    b = confidence_buckets([(0.05, True), (0.55, False), (1.0, True), (0.95, False)])
    assert len(b) == 10 and b[0]["count"] == 1 and b[5]["error_rate"] == 1.0
    assert b[9]["count"] == 2 and b[9]["accuracy"] == 0.5


def test_real_tuning_fixture_is_labelled_tuning_and_blocks_real_validation(tmp_path):
    sys.path.insert(0, str(ROOT / "experiments"))
    from observer_validation import fixture_mode

    res = fixture_mode(ROOT / "tests" / "fixtures" / "pokernow", tmp_path / "r.json")
    assert res["real_validation_status"] == "REAL VALIDATION: BLOCKED"
    assert res["by_role"]["tuning"]["frames"] == 1
    assert res["by_role"]["validation"]["frames"] == 0
    assert "INCLUDING tuning" in res["metrics_label"]


def _annotate_from_synthetic(tool, truths, role):
    for sid in tool.ids:
        t = truths[sid]
        seats = [{"status": "in hand", "stack": f"{s.stack:g}", "bet": f"{s.bet:g}"}
                 for s in t.seats]
        form = build_annotation(tool.empty(sid), status="complete", role=role,
                                hero_cards=t.hero_cards, board=" ".join(t.board) or "-",
                                dealer=str(t.dealer), pot=f"{t.pot:g}", seats=seats)
        tool.save(sid, form)


def test_session_annotation_end_to_end(tmp_path):
    tables = [table(), table(board=("Qs", "Jh", "4c"), pot=3.0)]
    rec, s = record(tmp_path / "s", frames_of(*tables, repeat=4))
    rec.finish()
    tool = SessionAnnotator(rec.path)
    assert tool.progress()["unreviewed"] == len(tool.ids) > 0
    draft = tool.prefill(tool.ids[-1])
    assert draft["prefilled_from_observer"] and draft["status"] == "unreviewed"
    assert draft["board"] == ["Qs", "Jh", "4c"]
    # ground truth = the synthetic table each frame was rendered from
    truths = {sid: tables[0] if int(sid) <= 4 else tables[1] for sid in tool.ids}
    _annotate_from_synthetic(tool, truths, "validation")
    tool.save(tool.ids[0], dict(tool.load(tool.ids[0]), status="skip"))
    prog = tool.progress()
    assert prog["skip"] == 1 and prog["annotated"] == len(tool.ids) - 1
    assert tool.next_unreviewed() is None
    sys.path.insert(0, str(ROOT / "experiments"))
    from observer_validation import fixture_mode

    res = fixture_mode(rec.path, tmp_path / "r.json")
    assert res["scored_annotations"] == len(tool.ids) - 1 and res["unscored"]["skip"] == 1
    rv = res["real_validation"]
    assert rv["status"] == "MEASURED" and rv["frames"] == len(tool.ids) - 1
    # synthetic frames: the observer reads its own renderer perfectly
    assert rv["metrics"]["full_state"]["accuracy"] > 0.9


def test_streamlit_annotate_mode(tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    rec, s = record(tmp_path / "sessions", frames_of(table(), table(board=("Qs", "Jh", "4c"),
                                                                    pot=3.0), repeat=4))
    rec.finish()
    at = AppTest.from_file(str(ROOT / "poker_alpha" / "ui" / "app.py"), default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("Developer").run()
    [r for r in at.sidebar.radio if r.label == "Input"][0].set_value("Annotate session").run()
    at.text_input(key="ann_root").set_value(str(tmp_path / "sessions")).run()
    assert not at.exception, at.exception
    assert any("Annotated 0 /" in str(m.value) for m in at.markdown)
    tool = SessionAnnotator(rec.path)
    sid = tool.ids[0]
    at.text_input(key=f"a_h1_{sid}").set_value("As")
    at.text_input(key=f"a_h2_{sid}").set_value("Kd")
    at.text_input(key=f"a_board_{sid}").set_value("-")
    at.selectbox(key=f"a_role_{sid}").set_value("validation")
    [b for b in at.button if b.label == "Save & next"][0].click().run()
    assert not at.exception, at.exception
    saved = json.loads((rec.path / "annotations" / f"{sid}.json").read_text())
    assert saved["hero_cards"] == ["As", "Kd"] and saved["board"] == []
    assert saved["status"] == "complete" and saved["role"] == "validation"
    assert any("Annotated 1 /" in str(m.value) for m in at.markdown)
    [b for b in at.button if b.label == "Prefill from observer readings"][0].click().run()
    assert not at.exception, at.exception
    assert any("verify every field" in str(w.value) for w in at.warning)


def test_sequence_metrics_on_synthetic_session(tmp_path):
    from poker_alpha.observer.sequence_metrics import (flicker_count, sequence_metrics,
                                                       truth_events)

    tables = [table(), table(bet1=2.0, stacks=(99.0, 96.0)),
              table(board=("Qs", "Jh", "4c"), pot=5.5),
              table(dealer=0, hero=("7c", "7d"))]
    from poker_alpha.observer.session import RetentionPolicy

    def mark_settled(i, s, rec):          # the 4th (settled) frame of each table
        if (i + 1) % 4 == 0:
            rec.mark(s)
    rec, s = record(tmp_path / "s", frames_of(*tables, repeat=4), policy=RetentionPolicy(()),
                    after=mark_settled)
    rec.finish()
    tool = SessionAnnotator(rec.path)
    # annotate the last (settled) frame of every block of 4 with its true table
    truths = {}
    for sid in tool.ids:
        n = int(sid)
        if n % 4 == 0:
            truths[sid] = tables[n // 4 - 1]
    for sid, t in truths.items():
        seats = [{"status": "in hand", "stack": f"{x.stack:g}", "bet": f"{x.bet:g}"}
                 for x in t.seats]
        tool.save(sid, build_annotation(tool.empty(sid), status="complete", role="validation",
                                        hero_cards=t.hero_cards, board=" ".join(t.board) or "-",
                                        dealer=str(t.dealer), pot=f"{t.pot:g}", seats=seats))
    m = sequence_metrics(rec.path)
    assert m["annotated_frames"] == 4 and m["intervals"] == 3
    ev = m["events"]
    assert ev["bet"]["true_positive"] == 1 and ev["bet"]["recall"] == 1.0
    assert ev["street"]["true_positive"] == 1 and ev["new_hand"]["true_positive"] == 1
    assert m["state_flicker"] == 0 and m["card_persistence_violations"] == 0
    assert m["streamed_frames"] == 16
    assert flicker_count(["a", "b", "a", "a"]) == 1 and flicker_count(["a", "b", "b"]) == 0
    # a dealer move alone is a new hand
    a = annotation_from_dict(dict(BASE, dealer_seat=0, annotated=["dealer_seat"]), "1",
                             Path("1.png"))
    b = annotation_from_dict(dict(BASE, dealer_seat=1, annotated=["dealer_seat"]), "2",
                             Path("2.png"))
    assert truth_events(a, b) == {("new_hand", None)}
