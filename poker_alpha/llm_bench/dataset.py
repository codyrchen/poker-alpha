"""Versioned benchmark dataset: canonical Kuhn decision states with
multiple semantically equivalent prompt templates (for paraphrase-
stability measurement). Deterministic; no provider responses included.
"""

from __future__ import annotations

from typing import Dict, List

BENCH_FORMAT = "pokeralpha.llmbench/v1"

CARD_NAMES = {0: "Jack", 1: "Queen", 2: "King"}

RULES_TEXT = (
    "Kuhn poker: three cards (Jack < Queen < King). Both players ante 1 "
    "chip and receive one private card. One betting round with actions "
    "'p' (pass: check, or fold when facing a bet) and 'b' (bet 1 chip, "
    "or call when facing a bet). If a bet is called, higher card wins "
    "the pot; if a bet is folded to, the bettor wins."
)

HISTORY_TEXT = {
    "": "You act first; no actions yet.",
    "p": "Your opponent passed (checked).",
    "b": "Your opponent bet 1 chip; 'p' folds, 'b' calls.",
    "pb": "You passed, then your opponent bet; 'p' folds, 'b' calls.",
}

#: Three semantically equivalent templates (paraphrase-stability probe).
TEMPLATES: Dict[str, str] = {
    "plain": (
        "{rules}\n\nYour private card: {card}. {history}\n"
        "Respond with ONLY a JSON object: "
        '{{"action": "p" or "b", "probabilities": {{"p": x, "b": y}}}} '
        "where the probabilities describe the mixed strategy you would "
        "play in this situation (they must sum to 1)."
    ),
    "terse": (
        "Game rules: {rules}\nCard: {card}. Situation: {history}\n"
        "Output JSON only: "
        '{{"action": <"p"|"b">, "probabilities": {{"p": <float>, "b": <float>}}}}. '
        "Probabilities = your intended mixed strategy, summing to 1."
    ),
    "narrative": (
        "You are playing a hand of Kuhn poker. {rules}\n"
        "You hold the {card}. {history}\n"
        "Decide how you would play this spot as a probability mix over "
        "'p' and 'b', then reply with nothing except the JSON object "
        '{{"action": "...", "probabilities": {{"p": ..., "b": ...}}}}.'
    ),
}

#: All 12 Kuhn information sets (card 0-2 x history where that card's
#: holder acts).
INFOSETS = [f"{c}{h}" for c in range(3) for h in ("", "p", "b", "pb")]


def build_dataset(seed: int = 0) -> dict:
    """Every infoset x template; deterministic ids."""
    items: List[dict] = []
    for key in INFOSETS:
        card, hist = int(key[0]), key[1:]
        for tname, tmpl in TEMPLATES.items():
            items.append({
                "id": f"kuhn-{key}-{tname}",
                "game": "kuhn",
                "infoset": key,
                "template": tname,
                "legal_actions": ["p", "b"],
                "prompt": tmpl.format(rules=RULES_TEXT,
                                      card=CARD_NAMES[card],
                                      history=HISTORY_TEXT[hist]),
            })
    return {"format": BENCH_FORMAT, "seed": seed, "n_items": len(items),
            "templates": list(TEMPLATES), "items": items}


__all__ = ["BENCH_FORMAT", "TEMPLATES", "INFOSETS", "build_dataset",
           "RULES_TEXT"]
