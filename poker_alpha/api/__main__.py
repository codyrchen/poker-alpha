"""Run the local PokerAlpha API:  python -m poker_alpha.api [--port N]

Binds 127.0.0.1 only. ``--port 0`` picks a free port and prints it as the
first line (``POKERALPHA_API_PORT=<n>``) so a desktop shell can attach.
"""

from __future__ import annotations

import argparse
import socket


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="loopback only; non-local hosts are refused")
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("refusing to bind a non-loopback host: this API is "
                         "local-only by design (see docs/local_api.md)")
    port = args.port
    if port == 0:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
    print(f"POKERALPHA_API_PORT={port}", flush=True)

    import uvicorn

    from .server import app

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
