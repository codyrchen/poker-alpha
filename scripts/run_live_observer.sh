#!/usr/bin/env bash
# Start the PokerAlpha live observer: doctor checks, then Streamlit on
# 127.0.0.1 with usage statistics off. Read-only; installs nothing.
#   ./scripts/run_live_observer.sh [--port 8501] [--force] [--no-capture-test]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ]; then PY="$VIRTUAL_ENV/bin/python";
  elif [ -x .venv/bin/python ]; then PY=.venv/bin/python;
  else PY=python3; fi
fi
if ! "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
  echo "Python >= 3.11 needed (found: $("$PY" --version 2>&1)). Set PYTHON=/path/to/python3.11+." >&2
  exit 1
fi
if ! "$PY" -c 'import poker_alpha' 2>/dev/null; then
  echo "poker_alpha is not importable with $PY. Install it first:" >&2
  echo "  $PY -m pip install -e '.[vision,ui]'" >&2
  exit 1
fi
exec "$PY" -m poker_alpha.live "$@"
