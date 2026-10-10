"""Phase 39 (final-trust): off-policy robustness vs fixed simple profiles.

Duplicate-deal, seat-swapped matches of a strategy artifact against scripted
opponents (calling station, over-folder, jam-heavy maniac, max-size
aggressor, tight-passive). Purpose: detect catastrophic exploitability or
bizarre behavior, NOT to prove equilibrium quality — beating simple bots is
weak evidence by design.

    python experiments/offpolicy_robustness.py --artifact <path> [--deals 5000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.abstraction.features import card_features  # noqa: E402
from poker_alpha.solver_config import V2_CONFIG  # noqa: E402
from poker_alpha.solvers.strategy_artifact import load_artifact  # noqa: E402
from poker_alpha.validation.crossplay import (calling_station, duplicate_match,  # noqa: E402
                                              table_policy)


def over_folder(game, state, legal):
    p = np.zeros(len(legal))
    p[legal.index("f") if "f" in legal else legal.index("c")] = 1.0
    return p


def jam_maniac(game, state, legal):
    p = np.zeros(len(legal))
    p[legal.index("a") if "a" in legal else legal.index("c")] = 1.0
    return p


def max_aggressor(game, state, legal):
    sized = [a for a in legal if a not in ("f", "c", "a")]
    pick = sized[-1] if sized else ("a" if "a" in legal else "c")
    p = np.zeros(len(legal))
    p[legal.index(pick)] = 1.0
    return p


def tight_passive(game, state, legal):
    me = game.current_player(state)
    hole = state.holes[me]
    f = card_features(hole, state.board)
    strong = (f.strength >= 4) if state.board else (f.strength >= 5)
    p = np.zeros(len(legal))
    if strong or "f" not in legal:
        p[legal.index("c")] = 1.0
    else:
        p[legal.index("f")] = 1.0
    return p


PROFILES = {"calling_station": calling_station, "over_folder": over_folder,
            "jam_maniac": jam_maniac, "max_aggressor": max_aggressor,
            "tight_passive": tight_passive}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--deals", type=int, default=5_000)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    game = V2_CONFIG.build_game()
    art = load_artifact(ROOT / args.artifact, game)
    hero = table_policy(lambda k: art.lookup(k))
    label = args.label or Path(args.artifact).stem
    out = {"format": "pokeralpha.offpolicy_robustness/v1",
           "artifact": args.artifact, "deals": args.deals,
           "note": ("duplicate deals, seat swap; beating scripted bots is a "
                    "sanity check, not equilibrium evidence"),
           "matches": {}}
    for i, (name, prof) in enumerate(PROFILES.items()):
        r = duplicate_match(game, hero, prof, args.deals, seed=12_000 + i)
        out["matches"][name] = r.to_dict()
        flag = "CATASTROPHIC" if r.bb_per_100 < -50 else (
            "negative" if r.bb_per_100 < 0 else "ok")
        print(f"{label} vs {name}: {r.bb_per_100:+8.1f} bb/100 "
              f"(CI {r.ci95_bb_per_100[0]:+.1f}..{r.ci95_bb_per_100[1]:+.1f}) [{flag}]",
              flush=True)
    dest = args.out or ROOT / f"results/validation/offpolicy_{label}.json"
    dest.write_text(json.dumps(out, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    main()
