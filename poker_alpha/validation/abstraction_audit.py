"""Encoder-independent abstraction measurement and perfect-recall audit.

Earlier infoset counts compared encoders across *separate* MCCFR runs, whose
different strategies sample different trajectories (and one run even used a
different bet-size menu). That does not measure compression. This module
instead:

1. generates ONE corpus of heads-up Hold'em decision states with a seeded,
   encoder-independent random legal policy,
2. deduplicates exact raw states and freezes the corpus,
3. passes every state through every encoder and counts distinct keys,
   overall and broken down by street, acting position, SPR bucket and
   betting depth,
4. inspects every abstract key with more than one raw member for
   perfect-recall violations.

Definitions
-----------
* *raw state* — full game state (both players' hole cards, board, history).
* *raw information state* — what the acting player observed: the raw
  encoder key (own hole cards, board, action history).
* An encoder is a **function of the raw information state** when it reads
  only those fields; then it can never produce more distinct keys than there
  are distinct raw keys on the same corpus (checked as an invariant).
* **Perfect recall in the abstract game** — every two raw states sharing an
  abstract key must share (a) the acting player's own action sequence,
  (b) the public action history, (c) the sequence of abstract keys the
  player held at each of their earlier decisions (what they "remember" of
  their own earlier private/public observations), and (d) the legal action
  set. (Different exact board cards inside one key are the intended effect
  of card abstraction, not a violation, as long as (c) holds.)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..games.holdem import HoldemGame, HoldemState, _tokens

STREET_NAMES = ("preflop", "flop", "turn", "river")
_BOARD_AT = (0, 3, 4, 5)


# -- corpus -----------------------------------------------------------------------

def generate_corpus(game: HoldemGame, hands: int, seed: int,
                    call_weight: float = 0.5, fold_weight: float = 0.15
                    ) -> List[HoldemState]:
    """Decision states from seeded random legal-policy rollouts.

    Policy (independent of any encoder): fold with weight ``fold_weight``
    when legal, check/call with ``call_weight``, the remaining weight spread
    uniformly over the legal bets/raises/all-in. Exact duplicate raw states
    are removed; order of first appearance is kept so the corpus is frozen
    for a given ``(game, hands, seed)``.
    """
    rng = np.random.default_rng(seed)
    seen = set()
    corpus: List[HoldemState] = []
    for _ in range(hands):
        s = game.deal(rng)
        while not game.is_terminal(s):
            if game.is_chance(s):
                s = game.sample_chance(s, rng)
                continue
            raw = (s.holes, s.board, s.streets)
            if raw not in seen:
                seen.add(raw)
                corpus.append(s)
            acts = game.legal_actions(s)
            w = []
            aggressive = [a for a in acts if a not in ("f", "c")]
            for a in acts:
                if a == "f":
                    w.append(fold_weight)
                elif a == "c":
                    w.append(call_weight)
                else:
                    w.append((1.0 - call_weight - (fold_weight if "f" in acts else 0.0))
                             / len(aggressive))
            p = np.array(w) / sum(w)
            s = game.next_state(s, acts[int(rng.choice(len(acts), p=p))])
    return corpus


def betting_depth(state: HoldemState) -> int:
    return sum(len(_tokens(st)) for st in state.streets)


def spr_bucket(game: HoldemGame, state: HoldemState,
               edges=(1.0, 3.0, 8.0)) -> int:
    from ..abstraction.cards import bucketize
    from ..abstraction.holdem import _street_start_spr

    street = len(state.streets) - 1
    return bucketize(_street_start_spr(game, state, street), edges)


# -- compression --------------------------------------------------------------------

@dataclass
class CompressionReport:
    states: int
    raw_keys: int
    keys: Dict[str, int]
    ratios: Dict[str, float]                  # raw keys / encoder keys
    by_street: Dict[str, Dict[str, int]]
    by_position: Dict[str, Dict[str, int]]
    by_spr_bucket: Dict[str, Dict[str, int]]
    by_depth: Dict[str, Dict[str, int]]
    invariant_violations: Dict[str, int]      # encoder -> #extra keys vs raw

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def measure_compression(game: HoldemGame, corpus: Sequence[HoldemState],
                        encoders: Mapping[str, object]) -> CompressionReport:
    """Distinct keys per encoder on the SAME states (raw = RawHoldemEncoder)."""
    from ..abstraction.holdem import RawHoldemEncoder

    raw_enc = RawHoldemEncoder()
    raw = [raw_enc.encode(game, s) for s in corpus]
    enc_keys = {name: [e.encode(game, s) for s in corpus]
                for name, e in encoders.items()}

    def group(fn) -> Dict[str, Dict[str, int]]:
        sets: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
        states: Dict[str, int] = defaultdict(int)
        for i, s in enumerate(corpus):
            g = fn(s)
            states[g] += 1
            sets[g]["raw"].add(raw[i])
            for name, ks in enc_keys.items():
                sets[g][name].add(ks[i])
        return {g: {"states": states[g], **{k: len(v) for k, v in d.items()}}
                for g, d in sorted(sets.items())}

    raw_n = len(set(raw))
    keys = {name: len(set(ks)) for name, ks in enc_keys.items()}
    violations = {}
    for name, ks in enc_keys.items():
        # A function of the raw information state maps each raw key to ONE key.
        img: Dict[str, set] = defaultdict(set)
        for r, k in zip(raw, ks):
            img[r].add(k)
        violations[name] = sum(len(v) - 1 for v in img.values())
    return CompressionReport(
        states=len(corpus), raw_keys=raw_n, keys=keys,
        ratios={n: raw_n / k for n, k in keys.items()},
        by_street=group(lambda s: STREET_NAMES[len(s.streets) - 1]),
        by_position=group(lambda s: "BTN/SB" if game.current_player(s) == 0 else "BB"),
        by_spr_bucket=group(lambda s: f"spr{spr_bucket(game, s)}"),
        by_depth=group(lambda s: f"depth{min(betting_depth(s), 8)}"
                       + ("+" if betting_depth(s) >= 8 else "")),
        invariant_violations=violations)


# -- perfect recall --------------------------------------------------------------------

def prior_decisions(game: HoldemGame, state: HoldemState
                    ) -> List[Tuple[HoldemState, str]]:
    """Earlier decision states of the player to act, with the action taken.

    Rebuilds every history prefix (with the board cards dealt by then) and
    keeps those where the same player was to act.
    """
    player = game.current_player(state)
    out = []
    for si, street in enumerate(state.streets):
        toks = _tokens(street)
        for j, tok in enumerate(toks):
            prefix = replace(
                state, board=state.board[:_BOARD_AT[si]],
                streets=state.streets[:si] + ("".join(toks[:j]),),
                folded=-1, all_in=False)
            if not game.is_chance(prefix) and game.current_player(prefix) == player:
                out.append((prefix, tok))
    return out


@dataclass
class CollisionCheck:
    key: str
    members: int
    own_actions_consistent: bool
    public_history_consistent: bool
    earlier_abstraction_consistent: bool
    legal_actions_consistent: bool
    raw_board_identical: bool            # informational (card abstraction merges boards)
    raw_hole_identical: bool             # informational

    @property
    def perfect_recall_ok(self) -> bool:
        return (self.own_actions_consistent and self.public_history_consistent
                and self.earlier_abstraction_consistent
                and self.legal_actions_consistent)

    def format(self) -> str:
        yn = lambda b: "yes" if b else "NO"  # noqa: E731
        return (f"abstract key: {self.key}\nmember count: {self.members}\n"
                f"own action history consistent: {yn(self.own_actions_consistent)}\n"
                f"public observation history consistent: {yn(self.public_history_consistent)}\n"
                f"earlier private abstraction consistent: {yn(self.earlier_abstraction_consistent)}\n"
                f"legal action set consistent: {yn(self.legal_actions_consistent)}")


@dataclass
class RecallAudit:
    encoder: str
    abstract_keys: int
    colliding_keys: int
    colliding_states: int
    violations: int
    violation_kinds: Dict[str, int]
    examples: List[CollisionCheck] = field(default_factory=list)

    @property
    def perfect_recall(self) -> bool:
        return self.violations == 0

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "examples"}
        d["perfect_recall"] = self.perfect_recall
        d["examples"] = [c.__dict__ for c in self.examples]
        return d


def audit_perfect_recall(game: HoldemGame, corpus: Sequence[HoldemState],
                         encoder, name: str, max_examples: int = 5,
                         max_members: int = 50) -> RecallAudit:
    """Inspect every abstract key with more than one raw-state member."""
    enc = lambda s: encoder.encode(game, s)  # noqa: E731
    groups: Dict[str, List[HoldemState]] = defaultdict(list)
    for s in corpus:
        groups[enc(s)].append(s)
    cache: Dict[tuple, str] = {}

    def key_of(s: HoldemState) -> str:
        k = (s.holes, s.board, s.streets)
        if k not in cache:
            cache[k] = enc(s)
        return cache[k]

    colliding = {k: v for k, v in groups.items() if len(v) > 1}
    kinds = {"own_actions": 0, "public_history": 0, "earlier_abstraction": 0,
             "legal_actions": 0}
    violations = 0
    examples: List[CollisionCheck] = []
    for key, members in colliding.items():
        members = members[:max_members]
        p = game.current_player(members[0])
        own = {tuple(t for _, t in prior_decisions(game, s)) for s in members}
        public = {s.streets for s in members}
        earlier = {tuple(key_of(pre) for pre, _ in prior_decisions(game, s))
                   for s in members}
        legal = {tuple(game.legal_actions(s)) for s in members}
        check = CollisionCheck(
            key=key, members=len(groups[key]),
            own_actions_consistent=len(own) == 1,
            public_history_consistent=len(public) == 1,
            earlier_abstraction_consistent=len(earlier) == 1,
            legal_actions_consistent=len(legal) == 1,
            raw_board_identical=len({s.board for s in members}) == 1,
            raw_hole_identical=len({tuple(sorted(s.holes[p])) for s in members}) == 1)
        if not check.perfect_recall_ok:
            violations += 1
            kinds["own_actions"] += not check.own_actions_consistent
            kinds["public_history"] += not check.public_history_consistent
            kinds["earlier_abstraction"] += not check.earlier_abstraction_consistent
            kinds["legal_actions"] += not check.legal_actions_consistent
            if len(examples) < max_examples:
                examples.append(check)
    if not examples:
        for key, members in list(colliding.items())[:max_examples]:
            p = game.current_player(members[0])
            examples.append(CollisionCheck(
                key, len(members), True, True, True, True,
                len({s.board for s in members}) == 1,
                len({tuple(sorted(s.holes[p])) for s in members}) == 1))
    return RecallAudit(name, len(groups), len(colliding),
                       sum(len(v) for v in colliding.values()), violations,
                       kinds, examples)


def generate_line_corpus(game: HoldemGame, streets: Tuple[str, ...], deals: int,
                         seed: int) -> List[HoldemState]:
    """Decision states that all follow ONE fixed action line, many deals.

    Holding the public action history constant isolates *card* abstraction:
    with exact histories in every key, only hole/board information can be
    merged. Deals on which the line is illegal (it never is for lines made
    of legal tokens) are skipped.
    """
    rng = np.random.default_rng(seed)
    out: List[HoldemState] = []
    seen = set()
    for _ in range(deals):
        s = game.deal(rng)
        ok = True
        for si, street in enumerate(streets):
            if si > 0:
                if not game.is_chance(s):
                    ok = False
                    break
                s = game.sample_chance(s, rng)
            for tok in _tokens(street):
                if game.is_terminal(s) or tok not in game.legal_actions(s):
                    ok = False
                    break
                s = game.next_state(s, tok)
            if not ok:
                break
        if not ok or game.is_terminal(s) or game.is_chance(s):
            continue
        raw = (s.holes, s.board, s.streets)
        if raw not in seen:
            seen.add(raw)
            out.append(s)
    return out
