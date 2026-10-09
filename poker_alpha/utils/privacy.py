"""Privacy checks for the running Streamlit server (Phase 72)."""

from __future__ import annotations

from typing import List

LOOPBACK = ("127.0.0.1", "localhost", "::1")


def streamlit_privacy_issues(get_option=None) -> List[str]:
    """Problems with the current Streamlit configuration ([] = fine)."""
    if get_option is None:
        try:
            from streamlit import config
        except ImportError:
            return []
        get_option = config.get_option
    issues = []
    addr = get_option("server.address")
    if not addr or str(addr) not in LOOPBACK:
        issues.append("Streamlit is listening on all network interfaces: other devices on your "
                      "network can open this page and see captured frames. Restart with "
                      "--server.address 127.0.0.1 (or ./scripts/run_live_observer.sh).")
    if get_option("browser.gatherUsageStats"):
        issues.append("Streamlit usage statistics are enabled (sent to Streamlit, not to "
                      "PokerAlpha; no frames). Restart with --browser.gatherUsageStats false "
                      "to turn them off.")
    return issues
