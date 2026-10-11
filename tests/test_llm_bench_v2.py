"""Phase II: LLM benchmark v2 - every mock failure mode must be detected."""

import json

import pytest

from poker_alpha.games.kuhn import KuhnPoker
from poker_alpha.llm_bench import v2


@pytest.fixture(scope="module")
def data():
    return v2.build_dataset()


@pytest.fixture(scope="module")
def results(data):
    return {m.name: v2.run(m(data, seed=1), data, repeats=2) for m in v2.MOCKS}


def test_dataset_shape_and_no_hidden_information(data):
    assert data["format"] == "pokeralpha.llmbench/v2"
    counts = {g: sum(1 for i in data["items"] if i["game"] == g) // 3 for g in data["games"]}
    assert counts == {"kuhn": 12, "leduc_r1": 18, "auction": 6, "signaling": 4}
    # prompts are functions of the infoset key only: two deals that differ only
    # in the opponent's card give identical prompts
    g = KuhnPoker()
    by_key = {}
    for _, s in g.chance_outcomes(g.root()):
        k = g.infoset_key(s)
        p = v2.render("kuhn", k, g.legal_actions(s), "natural")
        assert by_key.setdefault(k, p) == p
    for it in data["items"]:
        if it["game"] == "kuhn":
            own = v2.RANK[int(it["infoset"][0])]
            others = [r for c, r in v2.RANK.items() if c != int(it["infoset"][0])]
            body = it["prompt"].replace(v2.RULES["kuhn"], "")
            assert own in body and not any(o in body for o in others)


def test_perfect_mock_scores_reference(results):
    r = results["mock-perfect"]["per_game"]
    for g, d in r.items():
        assert d["parse_failure_rate"] == 0 and d["repeat_determinism"] == 1.0
        for rep, m in d["by_representation"].items():
            assert m["mean_tv_to_reference"] == pytest.approx(0.0, abs=1e-9)
            assert m["reference_completed_exploitability"] == pytest.approx(m["reference_exploitability"], abs=1e-9)
        assert all(x["mean_tv"] < 1e-9 for x in d["representation_instability"])


def test_first_action_mock_is_exploitable(results):
    r = results["mock-always-first"]["per_game"]
    for g, d in r.items():
        m = d["by_representation"]["json"]
        assert m["reference_completed_exploitability"] > m["reference_exploitability"] + 0.05, g


def test_random_mock_is_nondeterministic(results):
    dets = [d["repeat_determinism"] for d in results["mock-random"]["per_game"].values()]
    assert max(dets) < 1.0


def test_malformed_mock_failure_categories(results):
    total = {}
    for d in results["mock-malformed"]["per_game"].values():
        for k, v in d["failure_counts"].items():
            total[k] = total.get(k, 0) + v
    for cat in ("no_json", "illegal_action", "bad_probabilities", "bad_schema"):
        assert total.get(cat, 0) > 0, cat
    assert sum(d["inconsistent_action_rate"] for d in results["mock-malformed"]["per_game"].values()) > 0


def test_prompt_sensitive_mock_detected(results):
    for g, d in results["mock-prompt-sensitive"]["per_game"].items():
        assert max(x["mean_tv"] for x in d["representation_instability"]) > 0.1, g


def test_anthropic_provider_is_lazy_and_credential_free():
    import inspect
    src = inspect.getsource(v2)
    assert "sk-ant" not in src and "api_key=" not in src
