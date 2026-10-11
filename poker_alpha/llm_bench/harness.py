"""Benchmark harness: provider x dataset -> versioned result JSON.

Observable outputs only; every raw response, parse outcome and latency is
recorded. No results exist until a provider actually runs — the harness
never fabricates scores.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict

from .adapter import Provider
from .dataset import BENCH_FORMAT, build_dataset
from .metrics import evaluate_policy, paraphrase_instability
from .parser import ParseError, parse_response

RESULT_FORMAT = "pokeralpha.llmbench_result/v1"


def run_benchmark(provider: Provider, dataset: dict = None) -> dict:
    data = dataset or build_dataset()
    assert data["format"] == BENCH_FORMAT
    responses = []
    by_template: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    failures = defaultdict(int)
    for item in data["items"]:
        t0 = time.time()
        try:
            raw = provider.complete(item["prompt"])
        except Exception as exc:  # noqa: BLE001 - provider errors are data
            responses.append({"id": item["id"], "error": str(exc)})
            failures["provider_error"] += 1
            continue
        latency = time.time() - t0
        rec = {"id": item["id"], "infoset": item["infoset"],
               "template": item["template"], "raw": raw,
               "latency_s": round(latency, 4)}
        try:
            parsed = parse_response(raw, item["legal_actions"])
            rec["action"] = parsed.action
            rec["probabilities"] = parsed.probabilities
            if parsed.probabilities:
                by_template[item["template"]][item["infoset"]] = \
                    parsed.probabilities
        except ParseError as exc:
            rec["parse_error"] = exc.category
            failures[exc.category] += 1
        responses.append(rec)
    n = len(data["items"])
    result = {
        "format": RESULT_FORMAT,
        "provider": provider.name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_items": n,
        "failure_counts": dict(failures),
        "illegal_action_rate": failures.get("illegal_action", 0) / n,
        "responses": responses,
        "paraphrase_instability": paraphrase_instability(by_template),
        "policies_by_template": {
            t: evaluate_policy(pol) for t, pol in by_template.items()
            if pol},
    }
    return result


__all__ = ["RESULT_FORMAT", "run_benchmark"]
