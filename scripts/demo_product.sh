#!/usr/bin/env bash
# PokerAlpha — one-command product demo.
#
# Verifies the local pieces, starts the engine + Streamlit + the web app,
# and prints a menu of demo surfaces. Requires no PokerNow, no live play,
# no cloud credentials and no private data. Ctrl-C stops everything.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "No .venv found. Set up once with:"
  echo "  python3 -m venv .venv && .venv/bin/pip install -e '.[dev,ui,api,vision]'"
  exit 1
fi

echo "— PokerAlpha product demo —"
echo "[1/4] engine self-check…"
$PY - <<'EOF'
from poker_alpha.pipeline import load_solver, analyze, observe_manual
from poker_alpha.solver_config import RELEASE_STRATEGY
from poker_alpha.decision import DecisionConfig
from poker_alpha.demo_spots import DEMO_SPOTS

solver = load_solver(RELEASE_STRATEGY)
assert not hasattr(solver, "code"), f"strategy artifact unavailable: {solver}"
rep = analyze(observe_manual(DEMO_SPOTS["River, checked to hero — solver accepted"]),
              config=DecisionConfig(equity_simulations=300, rollout_simulations=300),
              solver=solver)
info = rep.details["solver"]
assert info["used"], info
print(f"    release strategy loaded; demo spot -> {rep.recommended} "
      f"(solver {info['confidence']})")
EOF

PIDS=()
cleanup() { echo; echo "stopping demo services…"; kill "${PIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT

echo "[2/4] local API on 127.0.0.1:8765…"
$PY -m poker_alpha.api >/tmp/pokeralpha_demo_api.log 2>&1 &
PIDS+=($!)

echo "[3/4] Streamlit app on 127.0.0.1:8501…"
$PY -m streamlit run poker_alpha/ui/app.py --server.headless true \
  >/tmp/pokeralpha_demo_streamlit.log 2>&1 &
PIDS+=($!)

WEB=""
if [ -d apps/web/node_modules ]; then
  echo "[4/4] web app on localhost:3000…"
  (cd apps/web && npm run dev >/tmp/pokeralpha_demo_web.log 2>&1) &
  PIDS+=($!)
  WEB="yes"
else
  echo "[4/4] web app skipped (run 'cd apps/web && npm install' to enable)"
fi

sleep 3
cat <<MENU

================= PokerAlpha demo surfaces =================
 1. Analyzer / abstention    http://localhost:8501   (Play)
 2. Session review + regret  http://localhost:8501   (Study -> Session Review)
 3. Hand replay + ranges     http://localhost:8501   (Study -> Hand Replay)
 4. Strategy explorer        http://localhost:8501   (Study -> Strategy Explorer)
 5. Trainer                  http://localhost:8501   (Study -> Trainer)
${WEB:+ 6. Web product frontend     http://localhost:3000}
${WEB:+ 7. Research / benchmarks    http://localhost:3000/benchmarks}
 8. Desktop overlay          cd apps/desktop && npm start
 9. Research story (CLI)     .venv/bin/python -m poker_alpha.demo
============================================================
 API: http://127.0.0.1:8765/docs   Logs: /tmp/pokeralpha_demo_*.log
 Everything is local; nothing is uploaded. Ctrl-C to stop.
MENU

wait
