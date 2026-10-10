"""Import hand histories into a session database and analyse it.

    python -m poker_alpha.session import hands.json --db session.sqlite
    python -m poker_alpha.session analyze --db session.sqlite [--session ID]
"""

from __future__ import annotations

import argparse
import sys

from ..decision import DecisionConfig
from ..history import load_hands
from . import SessionStore, analyze_session, import_hands


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m poker_alpha.session")
    sub = p.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import")
    imp.add_argument("path")
    imp.add_argument("--db", required=True)
    imp.add_argument("--rollouts", type=int, default=500)
    imp.add_argument("--equity-sims", type=int, default=1000)
    imp.add_argument("--seed", type=int, default=0)
    an = sub.add_parser("analyze")
    an.add_argument("--db", required=True)
    an.add_argument("--session", type=int, default=None)
    args = p.parse_args(argv)
    with SessionStore(args.db) as store:
        if args.cmd == "import":
            sid = import_hands(store, load_hands(args.path), source=args.path,
                               config=DecisionConfig(
                                   equity_simulations=args.equity_sims,
                                   rollout_simulations=args.rollouts,
                                   seed=args.seed))
            print(f"imported session {sid}")
            print(analyze_session(store, sid).format())
        else:
            sid = args.session or store.sessions()[-1]["id"]
            print(analyze_session(store, sid).format())
    return 0


if __name__ == "__main__":
    sys.exit(main())
