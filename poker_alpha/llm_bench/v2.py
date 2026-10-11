"""HiddenInfoBench-LLM v2 (Phase II, Program 16): multi-game decision dataset,
three prompt representations, strict parsing, exact metrics, mock agents.

Games (every item is generated from the acting player's INFORMATION-SET KEY
only, so no prompt can contain hidden information by construction; a test
checks that states differing only in hidden information yield identical
prompts):

  kuhn        all 12 infosets
  leduc_r1    round-1 Leduc infosets (3 ranks x 6 histories = 18)
  auction     AuctionDuel(3 values, 3 bids): 6 infosets
  signaling   2-type / 2-message costly signaling game: 4 infosets

Representations: ``json`` (structured state object), ``natural`` (prose) and
``compact`` (terse key=value). Semantically equivalent; differences in the
induced policy measure prompt instability.

Metrics per (provider, game, representation): parse-failure categories,
illegal-action rate, normalization failures, consistency (sampled action has
positive stated probability), exploitability (exact; unanswered infosets are
completed with the CFR+ reference and counted - "reference-completed
exploitability"), TV distance to the reference, repeat-determinism (each
item asked twice), and cross-representation instability.

No provider responses ship with the repository; real runs need credentials.
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional

from ..games.auction import AuctionDuel
from ..games.kuhn import KuhnPoker
from ..games.leduc import LeducPoker
from ..games.signaling import SignalingGame
from ..solvers.cfr_plus import CFRPlusSolver
from ..solvers.evaluation import exploitability
from .parser import ParseError, parse_response

FORMAT = "pokeralpha.llmbench/v2"
RESULT_FORMAT = "pokeralpha.llmbench_result/v2"
REPRESENTATIONS = ("json", "natural", "compact")
RANK = {0: "Jack", 1: "Queen", 2: "King"}


def _signaling():
    return SignalingGame((0.3, 0.7), ((1.0, -1.0), (-1.0, 1.0)), ((0.0, 0.6), (0.6, 0.0)), 2)


GAMES: Dict[str, Callable] = {"kuhn": KuhnPoker, "leduc_r1": LeducPoker,
                              "auction": lambda: AuctionDuel(3, 3), "signaling": _signaling}

RULES = {
    "kuhn": "Kuhn poker: cards Jack<Queen<King, ante 1, one round; 'p' = check or fold, 'b' = bet 1 or call.",
    "leduc_r1": ("Leduc poker, first betting round: six cards (two each of Jack, Queen, King), ante 1, private card "
                 "each; 'c' = check/call, 'r' = bet/raise 2 chips (max two raises), 'f' = fold. A public card follows."),
    "auction": ("Sealed-bid duel: your private value is 0-2; bid 0-2; higher bid wins (tie: nobody); "
                "your score = your surplus minus the opponent's surplus."),
    "signaling": ("Signaling game: you are the sender with a private type (0 = weak, 1 = strong) or the receiver who "
                  "sees only the message. Message 'm1' costs a weak sender 0.6, 'm0' costs a strong sender 0.6. "
                  "The receiver picks 'a0' or 'a1' and gains 1 if the action index matches the type, else loses 1."),
}


def _describe(game: str, key: str) -> dict:
    """Public description of an information set (only what the actor knows)."""
    if game == "kuhn":
        return {"your_card": RANK[int(key[0])], "history": key[1:] or "none", "you_act": "first" if len(key) in (1, 3) else "second"}
    if game == "leduc_r1":
        rank, _, hist = key.split("|", 2)
        return {"your_card": RANK[int(rank)], "history": hist or "none"}
    if game == "auction":
        return {"role": "first bidder" if key[0] == "A" else "second bidder", "your_value": int(key[1:])}
    if game == "signaling":
        return ({"role": "sender", "your_type": int(key[1:])} if key[0] == "S" else
                {"role": "receiver", "message_seen": f"m{key[1:]}"})
    raise ValueError(game)


def render(game: str, key: str, legal: List[str], rep: str) -> str:
    d = _describe(game, key)
    tail = ('Reply with ONLY JSON: {"action": <one of ' + json.dumps(legal) +
            '>, "probabilities": {<action>: <prob>, ...}} (your mixed strategy, summing to 1).')
    if rep == "json":
        return json.dumps({"game_rules": RULES[game], "state": d, "legal_actions": legal}, sort_keys=True) + "\n" + tail
    if rep == "natural":
        parts = ", ".join(f"{k.replace('_', ' ')} is {v}" for k, v in d.items())
        return f"{RULES[game]}\nIn this spot your {parts}. Legal actions: {', '.join(legal)}.\n{tail}"
    if rep == "compact":
        return f"{game}|" + ";".join(f"{k}={v}" for k, v in d.items()) + f"|legal={','.join(legal)}\n{tail}"
    raise ValueError(rep)


def infosets(game_name: str):
    g = GAMES[game_name]()
    out = {}

    def walk(s):
        if g.is_terminal(s):
            return
        if g.is_chance(s):
            for _, c in g.chance_outcomes(s):
                walk(c)
            return
        k = g.infoset_key(s)
        if game_name == "leduc_r1" and "/" in k.split("|", 2)[2]:
            return
        if game_name == "leduc_r1" and k.split("|")[1] != "?":
            return
        out.setdefault(k, g.legal_actions(s))
        for a in g.legal_actions(s):
            walk(g.next_state(s, a))
    walk(g.root())
    return out


def build_dataset() -> dict:
    items = []
    for game in GAMES:
        for key, legal in sorted(infosets(game).items()):
            for rep in REPRESENTATIONS:
                items.append({"id": f"{game}:{key}:{rep}", "game": game, "infoset": key, "representation": rep,
                              "legal_actions": list(legal), "prompt": render(game, key, list(legal), rep)})
    return {"format": FORMAT, "games": list(GAMES), "representations": list(REPRESENTATIONS),
            "n_items": len(items), "items": items}


_REF: Dict[str, dict] = {}


def reference(game: str) -> dict:
    if game not in _REF:
        s = CFRPlusSolver(GAMES[game]())
        s.train(2000 if game != "leduc_r1" else 400)
        _REF[game] = s.average_strategy()
    return _REF[game]


def policy_metrics(game: str, policy: Dict[str, Dict[str, float]]) -> dict:
    ref = reference(game)
    keys = set(infosets(game))
    full = dict(ref)
    full.update({k: v for k, v in policy.items() if k in ref})
    g = GAMES[game]()
    tv = [0.5 * sum(abs(policy[k].get(a, 0.0) - ref[k].get(a, 0.0)) for a in ref[k]) for k in policy if k in ref]
    return {"answered_infosets": len(set(policy) & keys), "benchmark_infosets": len(keys),
            "reference_completed_exploitability": exploitability(g, full),
            "reference_exploitability": exploitability(g, ref),
            "mean_tv_to_reference": (sum(tv) / len(tv)) if tv else None,
            "note": "unanswered infosets (and, for leduc_r1, all round-2 infosets) play the CFR+ reference"}


def run(provider, dataset: Optional[dict] = None, repeats: int = 2) -> dict:
    data = dataset or build_dataset()
    recs = []
    fails = defaultdict(lambda: defaultdict(int))
    policies = defaultdict(lambda: defaultdict(dict))     # (game, rep) -> infoset -> probs
    det = defaultdict(lambda: [0, 0])
    inconsistent = defaultdict(int)
    for it in data["items"]:
        answers = []
        for r in range(repeats):
            t0 = time.time()
            try:
                raw = provider.complete(it["prompt"])
            except Exception as e:  # noqa: BLE001 - provider errors are data
                fails[it["game"]]["provider_error"] += 1
                recs.append({"id": it["id"], "repeat": r, "error": str(e)})
                continue
            rec = {"id": it["id"], "repeat": r, "raw": raw, "latency_s": round(time.time() - t0, 4)}
            try:
                p = parse_response(raw, it["legal_actions"])
                rec["action"], rec["probabilities"] = p.action, p.probabilities
                answers.append(json.dumps([p.action, p.probabilities], sort_keys=True))
                if p.probabilities is None:
                    fails[it["game"]]["missing_probabilities"] += 1
                else:
                    if p.probabilities.get(p.action, 0.0) <= 0.0:
                        inconsistent[it["game"]] += 1
                        rec["inconsistent"] = True
                    if r == 0:
                        policies[(it["game"], it["representation"])][it["infoset"]] = p.probabilities
            except ParseError as e:
                rec["parse_error"] = e.category
                fails[it["game"]][e.category] += 1
            recs.append(rec)
        if len(answers) == repeats:
            det[it["game"]][0] += len(set(answers)) == 1
            det[it["game"]][1] += 1
    games = sorted({i["game"] for i in data["items"]})
    per_game = {}
    for g in games:
        n = sum(1 for i in data["items"] if i["game"] == g) * repeats
        reps = {}
        for rep in data["representations"]:
            pol = policies.get((g, rep), {})
            reps[rep] = policy_metrics(g, pol) if pol else {"answered_infosets": 0}
        inst = []
        rs = list(data["representations"])
        for i, a in enumerate(rs):
            for b in rs[i + 1:]:
                pa, pb = policies.get((g, a), {}), policies.get((g, b), {})
                common = set(pa) & set(pb)
                if common:
                    inst.append({"pair": [a, b], "infosets": len(common),
                                 "mean_tv": sum(0.5 * sum(abs(pa[k].get(x, 0) - pb[k].get(x, 0)) for x in set(pa[k]) | set(pb[k]))
                                                for k in common) / len(common)})
        per_game[g] = {"responses": n, "failure_counts": dict(fails[g]),
                       "illegal_action_rate": fails[g].get("illegal_action", 0) / n,
                       "parse_failure_rate": sum(v for k, v in fails[g].items() if k != "missing_probabilities") / n,
                       "inconsistent_action_rate": inconsistent[g] / n,
                       "repeat_determinism": (det[g][0] / det[g][1]) if det[g][1] else None,
                       "by_representation": reps, "representation_instability": inst}
    return {"format": RESULT_FORMAT, "provider": getattr(provider, "name", "unknown"), "repeats": repeats,
            "per_game": per_game, "responses": recs}


# --------------------------------------------------------------------------
# mock agents (each exhibits one known behaviour the benchmark must detect)
# --------------------------------------------------------------------------

class _Mock:
    name = "mock"

    def __init__(self, dataset=None, seed=0):
        self.by_prompt = {it["prompt"]: it for it in (dataset or build_dataset())["items"]}
        self.rng = random.Random(seed)
        self.calls = 0

    def item(self, prompt):
        return self.by_prompt[prompt]


class PerfectMock(_Mock):
    """Answers with the CFR+ reference mix (should score ~reference)."""
    name = "mock-perfect"

    def complete(self, prompt):
        it = self.item(prompt)
        probs = reference(it["game"])[it["infoset"]]
        act = max(probs, key=probs.get)
        return json.dumps({"action": act, "probabilities": probs})


class FirstActionMock(_Mock):
    name = "mock-always-first"

    def complete(self, prompt):
        it = self.item(prompt)
        a = it["legal_actions"][0]
        return json.dumps({"action": a, "probabilities": {x: float(x == a) for x in it["legal_actions"]}})


class RandomMock(_Mock):
    """Uniform mixes, but samples its stated action randomly (non-deterministic across repeats)."""
    name = "mock-random"

    def complete(self, prompt):
        it = self.item(prompt)
        L = it["legal_actions"]
        return json.dumps({"action": self.rng.choice(L), "probabilities": {x: 1.0 / len(L) for x in L}})


class MalformedMock(_Mock):
    """Cycles through failure modes: prose, illegal action, unnormalized, bad schema, inconsistent."""
    name = "mock-malformed"

    def complete(self, prompt):
        it = self.item(prompt)
        L = it["legal_actions"]
        self.calls += 1
        k = self.calls % 5
        if k == 0:
            return "I would probably bet here."
        if k == 1:
            return json.dumps({"action": "allin", "probabilities": {x: 1.0 / len(L) for x in L}})
        if k == 2:
            return json.dumps({"action": L[0], "probabilities": {x: 0.9 for x in L}})
        if k == 3:
            return json.dumps({"move": L[0]})
        return json.dumps({"action": L[0], "probabilities": {x: float(x == L[-1]) for x in L}})


class PromptSensitiveMock(_Mock):
    """Plays the reference under the JSON representation, a pure first action otherwise."""
    name = "mock-prompt-sensitive"

    def complete(self, prompt):
        it = self.item(prompt)
        if it["representation"] == "json":
            return PerfectMock.complete(self, prompt)
        return FirstActionMock.complete(self, prompt)


MOCKS = (PerfectMock, FirstActionMock, RandomMock, MalformedMock, PromptSensitiveMock)


# --------------------------------------------------------------------------
# provider plugin (no credentials shipped)
# --------------------------------------------------------------------------

class AnthropicProvider:
    """Claude via the official Anthropic SDK (imported lazily). Credentials are
    resolved by the SDK from the environment; nothing is embedded here.
    Model from ``POKERALPHA_LLM_MODEL`` (default ``claude-opus-5-5``).
    Server-side refusal fallback is enabled (``fallbacks="default"``); a final
    ``stop_reason == "refusal"`` is returned as an empty string so the harness
    records it as a parse failure rather than inventing an answer."""

    def __init__(self, model: Optional[str] = None, max_tokens: int = 16000) -> None:
        import os

        import anthropic  # noqa: F401 - lazy: only when this provider is used
        self.model = model or os.environ.get("POKERALPHA_LLM_MODEL", "claude-opus-5-5")
        self.client = anthropic.Anthropic()
        self.max_tokens = max_tokens
        self.name = f"anthropic:{self.model}"

    def complete(self, prompt: str) -> str:
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
        if resp.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in resp.content if b.type == "text")
