"""Determinism and resume audit (Phase 70).

Complements test_checkpoint.py (Kuhn/Leduc/default Hold'em resume) with the
release v2 configuration, and checks that results do not depend on Python's
per-process hash randomisation (set/dict-of-str iteration order), which an
in-process test cannot see.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from poker_alpha.solver_config import RELEASE_CONFIG
from poker_alpha.solvers import MCCFRSolver
from poker_alpha.solvers.digest import strategy_digest
from poker_alpha.solvers.serialize import load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def test_release_config_mccfr_resume_is_bit_identical(tmp_path):
    game = RELEASE_CONFIG.build_game()
    continuous = MCCFRSolver(game, seed=21)
    continuous.train(8)
    first = MCCFRSolver(RELEASE_CONFIG.build_game(), seed=21)
    first.train(5)
    resumed = load_checkpoint(save_checkpoint(first, tmp_path / "v2.npz"),
                              RELEASE_CONFIG.build_game())
    assert resumed.iterations == 5
    resumed.train(3)
    assert set(resumed.infosets) == set(continuous.infosets)
    assert strategy_digest(resumed.average_strategy(), None) == \
        strategy_digest(continuous.average_strategy(), None)


PROBE = r"""
import json, sys
sys.path.insert(0, "tests")
from poker_alpha.solver_config import RELEASE_CONFIG
from poker_alpha.solvers import MCCFRSolver
from poker_alpha.solvers.digest import strategy_digest
from test_golden_e2e import manual_cases, run_case

s = MCCFRSolver(RELEASE_CONFIG.build_game(), seed=2)
s.train(4)
out = {"mccfr": strategy_digest(s.average_strategy(), None),
       "signature": RELEASE_CONFIG.signature()}
obs, cfg, solver = manual_cases()["manual_flop_rollout"]
out["decision"] = run_case(obs, cfg, solver)
try:
    from poker_alpha.observer.pokernow import PokerNowStyleAdapter, default_layout
    from poker_alpha.observer.fusion import StateTracker
    from poker_alpha.observer.synthetic import SyntheticSeat, SyntheticTable, render_table
    cal = default_layout(2, 0)
    t = SyntheticTable(seats=[SyntheticSeat("h", 97.0), SyntheticSeat("v", 94.0, bet=6.0)],
                       dealer=1, hero_cards=("Td", "Tc"), board=("9s", "8h", "2d"),
                       pot=6.0, actor=0)
    img = render_table(t, cal)
    tr = StateTracker(cal, 0.5, 1.0)
    ad = PokerNowStyleAdapter(cal)
    for _ in range(3):
        tr.update(ad.read_frame(img))
    out["observer"] = repr(tr.to_observed_state())
except ImportError:
    out["observer"] = None
print(json.dumps(out, sort_keys=True, default=str))
"""


def _probe(hash_seed: str) -> dict:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    res = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env,
                         capture_output=True, text=True, check=True, timeout=600)
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_results_independent_of_python_hash_seed():
    a, b = _probe("0"), _probe("4242")
    assert a == b
