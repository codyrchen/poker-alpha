"""Phase II: HiddenInfoBench smoke (cheap tasks only)."""

import json

from poker_alpha import bench


def test_bench_manifest_fields_and_quick_tasks(tmp_path):
    ids = {t.id for t in bench.TASKS}
    assert len(ids) == len(bench.TASKS) >= 10
    for t in bench.TASKS:
        assert t.description and t.game and t.algorithm and t.budget and t.metric and t.expected
    doc = bench.run(["kuhn_cfr_plus", "auction_equilibrium", "confidence_calibration", "river_solver_parity",
                     "abstraction_refinement"], quick=True)
    assert doc["format"] == "pokeralpha.bench/v1"
    assert doc["status_counts"]["FAIL"] == 0 and doc["status_counts"]["error"] == 0
    assert not any("score" in k for k in doc)        # no aggregate score by design
    hw = doc["hardware"]
    assert "hostname" not in hw and "user" not in hw and hw["python"]
    assert bench.main(["--quick", "--tasks", "kuhn_cfr_plus", "--out", str(tmp_path / "b.json")]) == 0
    assert json.loads((tmp_path / "b.json").read_text())["tasks"][0]["status"] == "pass"
