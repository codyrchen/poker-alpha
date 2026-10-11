"""Local product API (pokeralpha.api/v1). ``python -m poker_alpha.api``."""

__all__ = ["create_app"]


def create_app():
    from .server import create_app as _c

    return _c()
