"""Phase 72: local-only behaviour checks."""

import re
from pathlib import Path

from poker_alpha.utils.privacy import streamlit_privacy_issues

ROOT = Path(__file__).resolve().parents[1]
NET = re.compile(r"^\s*(import|from)\s+(requests|urllib|http|socket|httpx|aiohttp|smtplib|"
                 r"ftplib|websocket)\b", re.M)


# The ONLY sanctioned network surface: the local product API, which binds
# 127.0.0.1 exclusively (poker_alpha/api/__main__.py refuses other hosts;
# tests/test_api.py and docs/local_api.md pin the policy). Everything else
# in the package stays network-free.
NET_ALLOWED = (ROOT / "poker_alpha" / "api",)


def test_no_network_code_in_the_package():
    for p in list((ROOT / "poker_alpha").rglob("*.py")) + list((ROOT / "experiments").glob("*.py")):
        if any(p.is_relative_to(a) for a in NET_ALLOWED):
            continue
        assert not NET.search(p.read_text()), p


def test_api_package_binds_loopback_only():
    src = (ROOT / "poker_alpha" / "api" / "__main__.py").read_text()
    assert 'refusing to bind a non-loopback host' in src
    assert 'host="127.0.0.1"' in src


def test_repository_streamlit_config_is_local_and_quiet():
    cfg = (ROOT / ".streamlit" / "config.toml").read_text()
    assert 'address = "127.0.0.1"' in cfg and "gatherUsageStats = false" in cfg


def test_privacy_issues_detected():
    opts = {"server.address": None, "browser.gatherUsageStats": True}
    issues = streamlit_privacy_issues(opts.get)
    assert len(issues) == 2 and "all network interfaces" in issues[0]
    ok = {"server.address": "127.0.0.1", "browser.gatherUsageStats": False}
    assert streamlit_privacy_issues(ok.get) == []


def test_captures_and_sessions_are_git_ignored():
    gi = (ROOT / ".gitignore").read_text()
    for pat in ("pokeralpha_captures/", "pokeralpha_sessions/", "pokeralpha_fixture_export*/"):
        assert pat in gi
