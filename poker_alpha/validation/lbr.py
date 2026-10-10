"""Restricted Local Best Response (final-trust project, Phases 17-18).

An APPROXIMATE LOWER BOUND on how exploitable a strategy is, following Lisy
& Bowling (2017), restricted to the abstract action menu:

* the LBR agent knows its own cards and tracks the opponent's range over
  all hands, Bayes-updating with the strategy's own action probabilities
  computed per hypothetical opponent hand;
* at its decisions it scores each legal action with the standard LBR
  heuristic — assume both players check/call to showdown afterwards:

      fold          ->  -own_contribution
      check/call    ->  wp * pot_after_call - contribution_after_call
      bet/raise     ->  fp * opp_contribution
                        + (1 - fp) * (wp_cont * pot_after_opp_call
                                       - own_contribution_after_bet)

  where wp is equity vs the current range, fp the range mass the strategy
  folds to this bet, wp_cont equity vs the continuing range (a raise in
  response is treated as a call for scoring only);
* it PLAYS the argmax action; the opponent plays the actual strategy with
  its actual cards. The reported number is the mean realized utility of the
  LBR seat — an unbiased estimate of the LBR policy's value, hence a lower
  bound (up to sampling error) on the true best-response value.

exploit_lb = (LBR value in seat 0 + LBR value in seat 1) / 2.

This is NEVER exact exploitability: it is restricted to the abstract menu,
uses a myopic value heuristic, and samples equities. Validation against
exact best-response values on reduced games: experiments/lbr_validation.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from ..poker.evaluator import evaluate_best_codes
from ..poker.ranges import COMBOS, NUM_COMBOS

# ---------------------------------------------------------------------------
# shared scoring
# ---------------------------------------------------------------------------


def score_actions(legal, contribs_after, opp_contrib_now, wp_call,
                  fold_prob, wp_cont, own_contrib_now) -> Dict[str, float]:
    """LBR heuristic value per legal action (see module docstring).

    ``contribs_after[a] = (own, opp)`` contributions after hero plays ``a``
    (opponent unchanged); for aggressive actions the opponent is assumed to
    call to hero's new level when continuing.
    """
    values: Dict[str, float] = {}
    for a in legal:
        own_after, _ = contribs_after[a]
        if a == "f":
            values[a] = -own_contrib_now
        elif a == "c":
            # after a call/check hero matches the opponent's level (HU,
            # equal stacks), so the showdown pot is own_after + opp_now
            pot = own_after + opp_contrib_now
            values[a] = wp_call * pot - own_after
        else:
            fp = fold_prob.get(a, 0.0)
            wpc = wp_cont.get(a, wp_call)
            pot_cont = 2.0 * own_after
            values[a] = (fp * opp_contrib_now
                         + (1.0 - fp) * (wpc * pot_cont - own_after))
    return values


# ---------------------------------------------------------------------------
# full-game Hold'em restricted LBR
# ---------------------------------------------------------------------------

@dataclass
class LBRResult:
    hands: int
    mean_bb_per_hand: float
    stderr_bb_per_hand: float

    @property
    def bb_per_100(self) -> float:
        return 100.0 * self.mean_bb_per_hand

    @property
    def ci95_bb_per_100(self):
        half = 196.0 * self.stderr_bb_per_hand
        return (self.bb_per_100 - half, self.bb_per_100 + half)

    def to_dict(self) -> dict:
        return {"hands": self.hands, "mean_bb_per_hand": self.mean_bb_per_hand,
                "stderr_bb_per_hand": self.stderr_bb_per_hand,
                "bb_per_100": self.bb_per_100,
                "ci95_bb_per_100": list(self.ci95_bb_per_100)}


class HoldemLBR:
    """Restricted LBR vs a strategy lookup on the release Hold'em game."""

    def __init__(self, game, lookup: Callable[[str], Optional[tuple]],
                 equity_samples: int = 160) -> None:
        self.game = game
        self.lookup = lookup            # key -> (probs dict, visits) | None
        self.equity_samples = equity_samples

    # -- strategy probabilities --------------------------------------------
    def _probs(self, state, legal) -> np.ndarray:
        hit = self.lookup(self.game.infoset_key(state))
        if hit is None:
            return np.full(len(legal), 1.0 / len(legal))
        p = np.array([hit[0].get(a, 0.0) for a in legal], dtype=float)
        s = p.sum()
        return p / s if s > 0 else np.full(len(legal), 1.0 / len(legal))

    def _per_combo_probs(self, state, legal, combo_ids, opp_seat) -> np.ndarray:
        """sigma(action | combo) for every live combo id: [n_combos, n_legal]."""
        from dataclasses import replace

        out = np.empty((len(combo_ids), len(legal)))
        for j, ci in enumerate(combo_ids):
            hole = (int(COMBOS[ci, 0]), int(COMBOS[ci, 1]))
            holes = ((state.holes[0], hole) if opp_seat == 1
                     else (hole, state.holes[1]))
            out[j] = self._probs(replace(state, holes=holes), legal)
        return out

    # -- equity vs range ----------------------------------------------------
    def _equity(self, hero, board, weights, rng) -> float:
        """MC equity of hero vs the weighted range on the (partial) board."""
        live_mask = weights > 0
        ids = np.nonzero(live_mask)[0]
        if len(ids) == 0:
            return 0.5
        w = weights[ids]
        cdf = np.cumsum(w)
        cdf /= cdf[-1]
        need = 5 - len(board)
        dead = set(hero) | set(board)
        deck = [c for c in range(52) if c not in dead]
        wins = 0.0
        n = self.equity_samples
        done = 0
        for _ in range(n * 3):          # rejection for combo/board collisions
            if done >= n:
                break
            ci = ids[int(np.searchsorted(cdf, rng.random(), side="right"))]
            oa, ob = int(COMBOS[ci, 0]), int(COMBOS[ci, 1])
            if oa in dead or ob in dead:
                continue
            if need:
                pick = rng.choice(len(deck), size=need, replace=False)
                run = [deck[i] for i in pick]
                if oa in run or ob in run:
                    continue
            else:
                run = []
            full = list(board) + run
            hv = evaluate_best_codes(list(hero) + full)
            ov = evaluate_best_codes([oa, ob] + full)
            wins += 1.0 if hv > ov else (0.5 if hv == ov else 0.0)
            done += 1
        return wins / max(done, 1)

    # -- one hand ------------------------------------------------------------
    def play_hand(self, lbr_seat: int, rng) -> float:
        from dataclasses import replace

        game = self.game
        state = game.sample_chance(game.root(), rng)
        hero = state.holes[lbr_seat]
        # opponent range: everything not colliding with hero's cards
        weights = np.ones(NUM_COMBOS)
        for c in hero:
            weights[(COMBOS[:, 0] == c) | (COMBOS[:, 1] == c)] = 0.0

        while not game.is_terminal(state):
            if game.is_chance(state):
                state = game.sample_chance(state, rng)
                for c in state.board:
                    weights[(COMBOS[:, 0] == c) | (COMBOS[:, 1] == c)] = 0.0
                continue
            player = game.current_player(state)
            legal = game.legal_actions(state)
            if player != lbr_seat:
                # strategy acts with its actual cards; Bayes-update the range
                probs = self._probs(state, legal)
                idx = int(rng.choice(len(legal), p=probs))
                ids = np.nonzero(weights > 0)[0]
                if len(ids):
                    per = self._per_combo_probs(state, legal, ids, player)
                    upd = weights.copy()
                    upd[ids] = weights[ids] * per[:, idx]
                    if upd.sum() > 0:
                        weights = upd
                state = game.next_state(state, legal[idx])
                continue

            # LBR decision
            own = state.contrib[lbr_seat]
            opp = state.contrib[1 - lbr_seat]
            wp_call = self._equity(hero, state.board, weights, rng)
            contribs_after = {}
            fold_prob = {}
            wp_cont = {}
            for a in legal:
                nxt = game.next_state(state, a)
                contribs_after[a] = (nxt.contrib[lbr_seat],
                                     nxt.contrib[1 - lbr_seat])
                if a not in ("f", "c"):
                    ids = np.nonzero(weights > 0)[0]
                    if len(ids) == 0:
                        fold_prob[a] = 0.0
                        continue
                    opp_legal = game.legal_actions(nxt)
                    per = self._per_combo_probs(nxt, opp_legal, ids, 1 - lbr_seat)
                    w = weights[ids]
                    w = w / w.sum()
                    if "f" in opp_legal:
                        fi = opp_legal.index("f")
                        fold_prob[a] = float((w * per[:, fi]).sum())
                        cont = weights.copy()
                        cont[ids] = weights[ids] * (1.0 - per[:, fi])
                    else:
                        fold_prob[a] = 0.0
                        cont = weights
                    if cont.sum() > 0 and fold_prob[a] < 0.999:
                        wp_cont[a] = self._equity(hero, state.board, cont, rng)
            values = score_actions(legal, contribs_after, opp, wp_call,
                                   fold_prob, wp_cont, own)
            best = max(values, key=values.get)
            state = game.next_state(state, best)

        u0 = game.utility(state)
        return u0 if lbr_seat == 0 else -u0

    def run(self, lbr_seat: int, hands: int, seed: int = 0) -> LBRResult:
        rng = np.random.default_rng(seed)
        utils = np.empty(hands)
        for i in range(hands):
            utils[i] = self.play_hand(lbr_seat, rng)
        return LBRResult(hands, float(utils.mean()),
                         float(utils.std(ddof=1) / np.sqrt(hands)))


# ---------------------------------------------------------------------------
# exact-subgame restricted LBR (validation of the method itself)
# ---------------------------------------------------------------------------

class SubgameLBR:
    """The same restricted-LBR algorithm on the exact reduced games, with
    EXACT equity (hand-value comparison / the game's equity table) instead of
    sampled runouts — used to validate the method against exact
    best-response values (experiments/lbr_validation.py)."""

    def __init__(self, game, strategy: Dict[str, Dict[str, float]]) -> None:
        from ..games.reduced_holdem import ReducedPreflopGame, RiverSubgame

        self.game = game
        self.strategy = strategy
        if isinstance(game, RiverSubgame):
            self.kind = "river"
        elif isinstance(game, ReducedPreflopGame):
            self.kind = "preflop"
        else:
            raise TypeError("SubgameLBR supports RiverSubgame / ReducedPreflopGame")

    # -- game plumbing -------------------------------------------------------
    def _stakes(self, state):
        """(contrib0, contrib1) including fixed stakes (half-pot / blinds)."""
        if self.kind == "river":
            c, _, _, _ = self.game._state(state.history)
            half = self.game.pot / 2.0
            return half + c[0], half + c[1]
        c, _, _ = self.game._contrib(state.history)
        return c[0], c[1]

    def _wp(self, hero_seat: int, hero, opp) -> float:
        if self.kind == "river":
            v0 = self.game._value[hero if hero_seat == 0 else opp]
            v1 = self.game._value[opp if hero_seat == 0 else hero]
            mine, theirs = (v0, v1) if hero_seat == 0 else (v1, v0)
            return 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
        eq = (self.game._equity[(hero, opp)] if hero_seat == 0
              else 1.0 - self.game._equity[(opp, hero)])
        return eq

    def _wp_range(self, hero_seat, hero, weights: Dict) -> float:
        tot = sum(weights.values())
        if tot <= 0:
            return 0.5
        return sum(w * self._wp(hero_seat, hero, o)
                   for o, w in weights.items()) / tot

    def _probs(self, state, legal) -> np.ndarray:
        probs = self.strategy.get(self.game.infoset_key(state))
        if not probs:
            return np.full(len(legal), 1.0 / len(legal))
        p = np.array([probs.get(a, 0.0) for a in legal])
        s = p.sum()
        return p / s if s > 0 else np.full(len(legal), 1.0 / len(legal))

    def _sub_holes(self, state, opp_seat: int, opp_hand):
        from dataclasses import replace

        holes = ((state.holes[0], opp_hand) if opp_seat == 1
                 else (opp_hand, state.holes[1]))
        return replace(state, holes=holes)

    # -- play ------------------------------------------------------------------
    def play_hand(self, lbr_seat: int, rng) -> float:
        game = self.game
        outcomes = game.chance_outcomes(game.root())
        probs = np.array([p for p, _ in outcomes])
        state = outcomes[int(rng.choice(len(outcomes), p=probs / probs.sum()))][1]
        hero = state.holes[lbr_seat]
        opp_seat = 1 - lbr_seat
        # initial range: conditional deal weights given hero's hand
        weights: Dict = {}
        for p, s in outcomes:
            if s.holes[lbr_seat] == hero:
                o = s.holes[opp_seat]
                weights[o] = weights.get(o, 0.0) + p

        while not game.is_terminal(state):
            player = game.current_player(state)
            legal = game.legal_actions(state)
            if player != lbr_seat:
                pr = self._probs(state, legal)
                idx = int(rng.choice(len(legal), p=pr))
                upd = {}
                for o, w in weights.items():
                    if w <= 0:
                        continue
                    po = self._probs(self._sub_holes(state, opp_seat, o), legal)
                    upd[o] = w * po[idx]
                if sum(upd.values()) > 0:
                    weights = upd
                state = game.next_state(state, legal[idx])
                continue

            own_now, opp_now = self._stakes(state)
            if lbr_seat == 1:
                own_now, opp_now = opp_now, own_now
            wp_call = self._wp_range(lbr_seat, hero, weights)
            contribs_after = {}
            fold_prob = {}
            wp_cont = {}
            for a in legal:
                nxt = game.next_state(state, a)
                c0, c1 = self._stakes(nxt)
                contribs_after[a] = ((c0, c1) if lbr_seat == 0 else (c1, c0))
                if a not in ("f", "c"):
                    opp_legal = game.legal_actions(nxt)
                    if "f" in opp_legal:
                        fi = opp_legal.index("f")
                        fp = 0.0
                        cont = {}
                        tot = sum(weights.values())
                        for o, w in weights.items():
                            if w <= 0:
                                continue
                            po = self._probs(self._sub_holes(nxt, opp_seat, o),
                                             opp_legal)
                            fp += w * po[fi]
                            cont[o] = w * (1.0 - po[fi])
                        fold_prob[a] = fp / tot if tot > 0 else 0.0
                        if sum(cont.values()) > 0:
                            wp_cont[a] = self._wp_range(lbr_seat, hero, cont)
                    else:
                        fold_prob[a] = 0.0
            values = score_actions(legal, contribs_after, opp_now, wp_call,
                                   fold_prob, wp_cont, own_now)
            state = game.next_state(state, max(values, key=values.get))

        u0 = game.utility(state)
        return u0 if lbr_seat == 0 else -u0

    def run(self, lbr_seat: int, hands: int, seed: int = 0) -> LBRResult:
        rng = np.random.default_rng(seed)
        utils = np.empty(hands)
        for i in range(hands):
            utils[i] = self.play_hand(lbr_seat, rng)
        return LBRResult(hands, float(utils.mean()),
                         float(utils.std(ddof=1) / np.sqrt(hands)))
