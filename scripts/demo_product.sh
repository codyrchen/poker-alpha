#!/usr/bin/env bash
# PokerAlpha — one-command product demo.
#
#   ./scripts/demo_product.sh             start services, wait until healthy,
#                                         open Streamlit + the web app
#   ./scripts/demo_product.sh --desktop   same, plus the desktop overlay
#   ./scripts/demo_product.sh --no-open   start services, print URLs only
#                                         (CI / headless / manual use)
#
# No PokerNow, no live play, no cloud credentials, no private data.
# Ctrl-C stops every service this script started.
set -euo pipefail
cd "$(dirname "$0")/.."

OPEN=1
DESKTOP=0
for arg in "$@"; do
  case "$arg" in
    --no-open) OPEN=0 ;;
    --desktop) DESKTOP=1 ;;
    *) echo "unknown flag: $arg (use --desktop / --no-open)"; exit 2 ;;
  esac
done

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "No .venv found. Set up once with:"
  echo "  python3 -m venv .venv && .venv/bin/pip install -e '.[dev,ui,api,vision]'"
  exit 1
fi

# Only macOS has `open`; everywhere else we just print URLs.
OPENER=""
if [ "$OPEN" = 1 ]; then
  case "$(uname -s)" in
    Darwin) OPENER="open" ;;
    Linux) command -v xdg-open >/dev/null 2>&1 && OPENER="xdg-open" ;;
  esac
fi

PIDS=()
cleanup() {
  trap - EXIT INT TERM
  echo
  echo "stopping demo services…"
  # children first (next dev lives under the npm wrapper), then parents
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do
    pkill -P "$pid" 2>/dev/null || true
  done
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

fail() { # <name> <log>
  echo "ERROR: $1 did not become healthy in time."
  echo "       log: $2"
  exit 1
}

wait_url() { # <url> <timeout_s>
  local url=$1 timeout=$2 t0
  t0=$(date +%s)
  until curl -sf -o /dev/null --max-time 2 "$url"; do
    if [ $(( $(date +%s) - t0 )) -ge "$timeout" ]; then
      return 1
    fi
    sleep 0.5
  done
}

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

echo "[2/4] local API on 127.0.0.1:8765…"
$PY -m poker_alpha.api >/tmp/pokeralpha_demo_api.log 2>&1 &
PIDS+=($!)
wait_url "http://127.0.0.1:8765/health" 60 \
  || fail "the API" /tmp/pokeralpha_demo_api.log
echo "    healthy (solver loaded)"

echo "[3/4] Streamlit app on 127.0.0.1:8501…"
$PY -m streamlit run poker_alpha/ui/app.py --server.headless true \
  >/tmp/pokeralpha_demo_streamlit.log 2>&1 &
PIDS+=($!)
wait_url "http://127.0.0.1:8501/_stcore/health" 90 \
  || fail "Streamlit" /tmp/pokeralpha_demo_streamlit.log
echo "    healthy"

WEB=""
if [ -d apps/web/node_modules ]; then
  echo "[4/4] web app on localhost:3000…"
  (cd apps/web && exec npm run dev >/tmp/pokeralpha_demo_web.log 2>&1) &
  PIDS+=($!)
  wait_url "http://localhost:3000" 120 \
    || fail "the web app" /tmp/pokeralpha_demo_web.log
  echo "    healthy"
  WEB="yes"
else
  echo "[4/4] web app skipped (run 'cd apps/web && npm install' to enable)"
fi

if [ "$DESKTOP" = 1 ]; then
  if [ -d apps/desktop/node_modules ]; then
    echo "[+] desktop overlay…"
    (cd apps/desktop && exec npm start >/tmp/pokeralpha_demo_desktop.log 2>&1) &
    PIDS+=($!)
  else
    echo "[+] desktop skipped (run 'cd apps/desktop && npm install' first)"
  fi
fi

if [ -n "$OPENER" ]; then
  "$OPENER" "http://localhost:8501" 2>/dev/null || true
  [ -n "$WEB" ] && { "$OPENER" "http://localhost:3000" 2>/dev/null || true; }
fi

cat <<MENU

================= PokerAlpha demo surfaces =================
 1. Analyzer / abstention    http://localhost:8501   (Play)
 2. Session review + regret  http://localhost:8501   (Study -> Session Review)
 3. Hand replay + ranges     http://localhost:8501   (Study -> Hand Replay)
 4. Strategy explorer        http://localhost:8501   (Study -> Strategy Explorer)
 5. Trainer                  http://localhost:8501   (Study -> Trainer)
${WEB:+ 6. Web product frontend     http://localhost:3000}
${WEB:+ 7. 90-second tour           http://localhost:3000/tour}
${WEB:+ 8. Watch a Solver Learn     http://localhost:3000/learn}
${WEB:+ 9. Arena (play it)          http://localhost:3000/arena}
${WEB:+10. Research Lab             http://localhost:3000/lab}
${WEB:+11. Decision Surface         http://localhost:3000/surface}
${WEB:+12. Trust Map                http://localhost:3000/trust}
${WEB:+13. Engineering story        http://localhost:3000/engineering}
14. Desktop overlay          ./scripts/demo_product.sh --desktop
15. Research story (CLI)     .venv/bin/python -m poker_alpha.demo
============================================================
 API: http://127.0.0.1:8765/docs   Logs: /tmp/pokeralpha_demo_*.log
 Everything is local; nothing is uploaded. Ctrl-C to stop.
MENU

wait
