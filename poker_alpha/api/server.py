"""Local PokerAlpha product API (FastAPI).

Security / privacy model for v0.1 (see docs/local_api.md):

* binds 127.0.0.1 only — this is a LOCAL engine for local frontends;
* CORS is restricted to explicit local development origins;
* no training endpoints, no filesystem access, no shell, no telemetry;
* every request body is a poker state or a hand-history document — files
  never leave the machine.

Public deployment would require authentication, rate limiting and input
hardening that v0.1 deliberately does not claim.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..decision import DecisionConfig
from ..solver_config import (RELEASE_CONFIDENCE, RELEASE_ITERATIONS,
                             RELEASE_STRATEGY)
from . import schemas as S

ROOT = Path(__file__).resolve().parents[2]

LOCAL_ORIGINS = [
    "http://localhost:3000", "http://127.0.0.1:3000",   # Next.js dev
    "http://localhost:8501", "http://127.0.0.1:8501",   # Streamlit
    # The Electron overlay loads its renderer from file://, whose Origin is
    # the literal string "null". The real security boundary is the loopback
    # bind (the server is unreachable off-machine); CORS here is
    # defense-in-depth for browser tabs, not the perimeter.
    "null",
]


@lru_cache(maxsize=1)
def _solver():
    """The release strategy, loaded once per process."""
    from ..pipeline import load_solver

    return load_solver(str(ROOT / RELEASE_STRATEGY))


def _config(settings: S.AnalysisSettings) -> DecisionConfig:
    solver, unavailable = None, None
    if settings.use_solver:
        prov = _solver()
        if hasattr(prov, "code"):
            unavailable = prov
        else:
            solver = prov
    return DecisionConfig(equity_simulations=settings.equity_simulations,
                          rollout_simulations=settings.rollout_simulations,
                          seed=settings.seed, solver=solver,
                          solver_unavailable=unavailable)


def _to_response(report, input_source: str = "manual") -> S.AnalyzeResponse:
    """DecisionReport -> wire format, through the Play Mode view models so
    the semantics stay single-sourced."""
    from ..ui.viewmodel import recommendation_vm, why_vm

    vm = recommendation_vm(report)
    why = why_vm(report)
    info = report.details.get("solver", {}) if report.details else {}
    by_label = {c.label: c for c in report.candidates}
    resp = S.AnalyzeResponse(
        state_summary=report.state_summary,
        actions=[S.ActionOut(
            label=a.label,
            kind=by_label[a.label].kind if a.label in by_label else "",
            amount_to=by_label[a.label].amount_to if a.label in by_label else None,
            frequency=a.frequency, ev_bb=a.ev_bb, ev_se_bb=a.ev_se_bb,
            source=a.source, recommended=a.recommended)
            for a in vm.actions],
        recommended=report.recommended, recommended_tag=vm.recommended_tag,
        mix_meaning=report.mix_meaning, method=report.method,
        source=vm.source_kind if report.method != "none" else "none",
        solver_state=vm.solver_state,
        decision_confidence=report.confidence,
        ev_is_decision_basis=vm.ev_is_decision_basis,
        equity=report.hero_equity, equity_se=report.hero_equity_se,
        pot_bb=report.pot_bb, to_call_bb=report.to_call_bb,
        pot_odds=report.pot_odds,
        effective_stack_bb=report.effective_stack_bb,
        warnings=list(report.warnings),
        provenance=S.ProvenanceOut(
            input_source=input_source,
            gate_reasons=list(info.get("reasons", ())),
            gate_signals={k: v for k, v in info.items()
                          if isinstance(v, (int, float, type(None)))
                          and k not in ("used",)},
            source_cascade=list(report.details.get("source_cascade", ())),
            ev_note=why.ev_estimates_note),
    )
    if vm.abstention is not None:
        resp.abstention = S.AbstentionOut(
            withheld=vm.abstention.withheld, title=vm.abstention.title,
            reasons=list(vm.abstention.reasons),
            reason_codes=list(vm.abstention.reason_codes),
            fallback=vm.abstention.fallback)
    if vm.refusal is not None:
        resp.refusal = S.RefusalOut(codes=list(vm.refusal.codes),
                                    message=vm.refusal.message)
    return resp


def create_app() -> FastAPI:
    app = FastAPI(title="PokerAlpha local API", version="0.1.0",
                  description="Local decision-support engine. Analysis only: "
                              "it never clicks, bets or controls a client.")
    app.add_middleware(CORSMiddleware, allow_origins=LOCAL_ORIGINS,
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.get("/health", response_model=S.HealthResponse)
    def health() -> S.HealthResponse:
        return S.HealthResponse(solver_loaded=not hasattr(_solver(), "code"))

    @app.get("/version")
    def version() -> dict:
        return {"api": S.API_VERSION, "app": "0.1.0"}

    @app.get("/release/info", response_model=S.ReleaseInfo)
    def release_info() -> S.ReleaseInfo:
        path = ROOT / RELEASE_STRATEGY
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        prov = _solver()
        cfg_sig = prov.game.solver_config.signature() \
            if not hasattr(prov, "code") else "unavailable"
        enc = type(prov.game.encoder).__name__ \
            if not hasattr(prov, "code") else "unavailable"
        return S.ReleaseInfo(
            strategy_path=RELEASE_STRATEGY, strategy_sha256=sha,
            confidence_path=RELEASE_CONFIDENCE,
            iterations=RELEASE_ITERATIONS, config_signature=cfg_sig,
            encoder=enc,
            disclaimer=("EXPERIMENTAL: an MCCFR average strategy on an "
                        "imperfect-recall abstraction; measurably exploitable "
                        "in the full game; not an equilibrium, not GTO."))

    @app.get("/demo-spots", response_model=list[S.DemoSpot])
    def demo_spots() -> list[S.DemoSpot]:
        from ..ui.play import DEMO_SPOTS

        return [S.DemoSpot(name=k, state=v) for k, v in DEMO_SPOTS.items()]

    @app.post("/analyze", response_model=S.AnalyzeResponse)
    def analyze_state(req: S.AnalyzeRequest) -> S.AnalyzeResponse:
        from ..pipeline import analyze, observe_manual

        try:
            obs = observe_manual(req.state)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, f"invalid state: {exc}")
        report = analyze(obs, config=_config(req.settings))
        return _to_response(report)

    @app.post("/what-if", response_model=S.WhatIfResponse)
    def what_if(req: S.WhatIfRequest) -> S.WhatIfResponse:
        from ..analysis.whatif import WHATIF_VARIABLES, compare_whatif

        if req.variable not in WHATIF_VARIABLES:
            raise HTTPException(422, f"unknown variable {req.variable!r}; "
                                     f"supported: {list(WHATIF_VARIABLES)}")
        try:
            variants, rejected = compare_whatif(
                req.state, req.variable, req.values,
                config=_config(req.settings))
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, f"invalid base state: {exc}")
        return S.WhatIfResponse(
            variable=req.variable,
            variants=[S.WhatIfVariant(
                value=v.value, solver_state=v.solver_state, method=v.method,
                confidence=v.confidence, strategy=v.strategy_text(),
                actions=[S.ActionOut(label=lbl, kind="", frequency=f)
                         for lbl, f in v.top_actions])
                for v in variants],
            rejected=rejected)

    @app.post("/session/review", response_model=S.SessionReviewResponse)
    def session_review(req: S.SessionReviewRequest) -> S.SessionReviewResponse:
        import json as _json

        from ..analysis import (detect_patterns, review_hands,
                                review_to_json, summarize)
        from ..history import parse_hands

        try:
            hands = parse_hands(req.hands)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, f"invalid hand history: {exc}")
        if len(hands) > 500:
            raise HTTPException(422, "at most 500 hands per request")
        rev = review_hands(hands, config=_config(req.settings))
        s = summarize(rev.records, rev.hands, rev.warnings)
        payload = _json.loads(review_to_json(rev, s, detect_patterns(rev.records)))
        return S.SessionReviewResponse(review=payload)

    @app.get("/strategy/preflop", response_model=S.PreflopMatrixResponse)
    def preflop(node: str = "BTN first action") -> S.PreflopMatrixResponse:
        from ..analysis.explorer import (PREFLOP_NODES, matrix_stats,
                                         preflop_matrix)

        if node not in PREFLOP_NODES:
            raise HTTPException(404, f"unknown node; known: {list(PREFLOP_NODES)}")
        prov = _solver()
        if hasattr(prov, "code"):
            raise HTTPException(503, "strategy artifact unavailable")
        m = preflop_matrix(prov, node)
        return S.PreflopMatrixResponse(
            node=node, nodes=list(PREFLOP_NODES),
            cells=[[S.MatrixCell(hand=c.hand, combo=c.combo, status=c.status,
                                 dominant=c.dominant,
                                 dominant_freq=c.dominant_freq,
                                 mix=[[a, p] for a, p in c.mix],
                                 visits=c.visits, reasons=list(c.reasons),
                                 notes=list(c.notes)) for c in row]
                   for row in m],
            stats=matrix_stats(m))

    @app.get("/strategy/preflop/{hand}", response_model=S.MatrixCell)
    def preflop_hand(hand: str, node: str = "BTN first action") -> S.MatrixCell:
        from ..analysis.explorer import PREFLOP_NODES, hand_classes, lookup_cell

        if node not in PREFLOP_NODES:
            raise HTTPException(404, f"unknown node; known: {list(PREFLOP_NODES)}")
        valid = {c for row in hand_classes() for c in row}
        if hand not in valid:
            raise HTTPException(404, f"{hand!r} is not a canonical hand class")
        prov = _solver()
        if hasattr(prov, "code"):
            raise HTTPException(503, "strategy artifact unavailable")
        c = lookup_cell(prov, hand, node)
        return S.MatrixCell(hand=c.hand, combo=c.combo, status=c.status,
                            dominant=c.dominant, dominant_freq=c.dominant_freq,
                            mix=[[a, p] for a, p in c.mix], visits=c.visits,
                            reasons=list(c.reasons), notes=list(c.notes))

    return app


app = create_app()

__all__ = ["app", "create_app"]
