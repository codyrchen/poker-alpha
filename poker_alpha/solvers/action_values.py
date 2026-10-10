"""Counterfactual action values under a strategy profile (final-trust Phase 6).

For every infoset of a small enumerable game, compute the normalized
counterfactual value of each action under a profile (both players follow
``profile``, uniform at unknown keys):

    q(I, a) = sum_{h in I} pi_chance(h) * pi_{-i}(h) * EV_i(h a | profile)
              -------------------------------------------------------------
              sum_{h in I} pi_chance(h) * pi_{-i}(h)

q is expressed for the acting player, in the game's chip units, per
(counterfactual) visit. From q the decision-consequence metrics follow:

* action margin  = best q − second-best q;
* EV regret of a policy p at I = max_a q(I,a) − Σ_a p(a) q(I,a);
* wrong-best-action = argmax_a p(a) != argmax_a q(I,a).

Exact (full tree enumeration): small games only.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple


def cf_action_values(game, profile: Mapping[str, Mapping[str, float]]
                     ) -> Dict[str, dict]:
    """``{key: {"q": {action: value}, "reach": cf_reach, "actions": [...]}}``."""
    num: Dict[str, Dict[str, float]] = {}
    den: Dict[str, float] = {}
    acts_of: Dict[str, Tuple[str, ...]] = {}

    def walk(state, reach_chance: float, reach0: float, reach1: float) -> float:
        if game.is_terminal(state):
            return game.utility(state)
        if game.is_chance(state):
            total = 0.0
            for p, child in game.chance_outcomes(state):
                total += p * walk(child, reach_chance * p, reach0, reach1)
            return total
        player = game.current_player(state)
        key = game.infoset_key(state)
        actions = game.legal_actions(state)
        probs = profile.get(key)
        child_ev = []
        for a in actions:
            pa = probs.get(a, 0.0) if probs else 1.0 / len(actions)
            child_ev.append(walk(game.next_state(state, a),
                                 reach_chance,
                                 reach0 * (pa if player == 0 else 1.0),
                                 reach1 * (pa if player == 1 else 1.0)))
        node_ev = 0.0
        for a, ev in zip(actions, child_ev):
            pa = probs.get(a, 0.0) if probs else 1.0 / len(actions)
            node_ev += pa * ev
        cf = reach_chance * (reach1 if player == 0 else reach0)
        if cf > 0.0:
            sign = 1.0 if player == 0 else -1.0
            bucket = num.setdefault(key, {})
            for a, ev in zip(actions, child_ev):
                bucket[a] = bucket.get(a, 0.0) + cf * sign * ev
            den[key] = den.get(key, 0.0) + cf
            acts_of.setdefault(key, tuple(actions))
        return node_ev

    walk(game.root(), 1.0, 1.0, 1.0)
    out: Dict[str, dict] = {}
    for key, bucket in num.items():
        d = den[key]
        out[key] = {"q": {a: v / d for a, v in bucket.items()},
                    "reach": d, "actions": list(acts_of[key])}
    return out


def policy_regret(q: Mapping[str, float],
                  policy: Optional[Mapping[str, float]]) -> float:
    """max_a q − Σ_a p(a) q(a); uniform policy if None."""
    best = max(q.values())
    if policy:
        mix = sum(policy.get(a, 0.0) * v for a, v in q.items())
        tot = sum(policy.get(a, 0.0) for a in q)
        mix = mix / tot if tot > 0 else sum(q.values()) / len(q)
    else:
        mix = sum(q.values()) / len(q)
    return best - mix


def action_margin(q: Mapping[str, float]) -> float:
    vals = sorted(q.values(), reverse=True)
    return vals[0] - vals[1] if len(vals) > 1 else 0.0
