"""Importer for the ``pokeralpha-lines/v1`` plain-text session format.

A deliberately small line-based format that people can type or script by
hand — the fixture-backed REFERENCE importer demonstrating the protocol
(detect/parse -> canonical events). Example (see
``tests/fixtures/imports/simple_lines_session.txt``):

    # pokeralpha-lines/v1
    hand h1 seats=2 dealer=0 sb=0.5 bb=1 hero=0 stacks=100,100
    deal 0 Ah Kd
    raise 0 2.5
    call 1
    flop Qs 7d 2c
    check 1
    bet 0 3.5
    fold 1

Rules: one command per line; ``#`` comments and blank lines ignored;
``call`` takes no amount (the engine computes it); ``bet``/``raise`` take
the raise-to amount; streets are ``flop c c c`` / ``turn c`` / ``river c``;
``show <seat> <c1> <c2>`` records a showdown. Every hand must start with a
``hand`` line.
"""

from __future__ import annotations

from typing import List

name = "pokeralpha-lines/v1"

_HEADER = "# pokeralpha-lines/v1"


def detect(text: str) -> bool:
    head = text.lstrip().splitlines()[:1]
    return bool(head) and head[0].strip() == _HEADER


def _err(i: int, line: str, msg: str):
    from . import ImportFormatError

    return ImportFormatError(f"line {i + 1} ({line.strip()!r}): {msg}")


def parse(text: str) -> List[list]:
    from ..events import event_from_dict

    hands: List[list] = []
    current: List[dict] = []

    def flush():
        if current:
            hands.append([event_from_dict(e) for e in current])

    for i, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cmd = parts[0].lower()
        try:
            if cmd == "hand":
                flush()
                current = []
                kv = dict(p.split("=", 1) for p in parts[2:])
                stacks = [float(x) for x in kv["stacks"].split(",")]
                n = int(kv["seats"])
                if len(stacks) != n:
                    raise _err(i, raw, f"{n} seats but {len(stacks)} stacks")
                dealer = int(kv["dealer"])
                sb = float(kv["sb"])
                bb = float(kv["bb"])
                hero = int(kv["hero"])
                current.append({
                    "type": "HandStarted", "hand_id": parts[1],
                    "num_seats": n, "dealer": dealer, "small_blind": sb,
                    "big_blind": bb, "hero_seat": hero,
                    "seats": [{"seat": s, "name": f"p{s}",
                               "stack": stacks[s]} for s in range(n)]})
                sb_seat = dealer if n == 2 else (dealer + 1) % n
                bb_seat = (sb_seat + 1) % n
                current.append({"type": "BlindPosted", "seat": sb_seat,
                                "amount": sb, "blind": "sb"})
                current.append({"type": "BlindPosted", "seat": bb_seat,
                                "amount": bb, "blind": "bb"})
            elif not current:
                raise _err(i, raw, "action before any 'hand' line")
            elif cmd == "deal":
                current.append({"type": "CardDealt", "seat": int(parts[1]),
                                "cards": [parts[2], parts[3]]})
            elif cmd == "flop":
                current.append({"type": "FlopDealt",
                                "cards": [parts[1], parts[2], parts[3]]})
            elif cmd == "turn":
                current.append({"type": "TurnDealt", "card": parts[1]})
            elif cmd == "river":
                current.append({"type": "RiverDealt", "card": parts[1]})
            elif cmd == "fold":
                current.append({"type": "PlayerFolded", "seat": int(parts[1])})
            elif cmd == "check":
                current.append({"type": "PlayerChecked", "seat": int(parts[1])})
            elif cmd == "call":
                current.append({"type": "PlayerCalled", "seat": int(parts[1])})
            elif cmd == "bet":
                current.append({"type": "PlayerBet", "seat": int(parts[1]),
                                "to": float(parts[2])})
            elif cmd == "raise":
                current.append({"type": "PlayerRaised", "seat": int(parts[1]),
                                "to": float(parts[2])})
            elif cmd == "allin":
                current.append({"type": "PlayerAllIn", "seat": int(parts[1]),
                                "to": float(parts[2])})
            elif cmd == "show":
                current.append({"type": "Showdown", "seat": int(parts[1]),
                                "cards": [parts[2], parts[3]]})
            else:
                raise _err(i, raw, f"unknown command {cmd!r}")
        except (IndexError, KeyError, ValueError) as exc:
            from . import ImportFormatError

            if isinstance(exc, ImportFormatError):
                raise
            raise _err(i, raw, str(exc)) from exc
    flush()
    if not hands:
        from . import ImportFormatError

        raise ImportFormatError("no hands found")
    return hands
