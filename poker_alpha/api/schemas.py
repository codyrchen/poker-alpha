"""pokeralpha.api/v1 — versioned request/response contract.

Frontends depend on THIS module, never on internal Python object
serialization. The semantics mirror the Play Mode view models: solver
frequencies are a distribution (highest-frequency action, not a single
prescription); EVs are flagged as decision basis or as separate rollout
estimates; abstention and refusal are first-class states.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

API_VERSION = "pokeralpha.api/v1"


class AnalysisSettings(BaseModel):
    equity_simulations: int = Field(2000, ge=50, le=20000)
    rollout_simulations: int = Field(1000, ge=0, le=10000)
    seed: int = 0
    use_solver: bool = True


class AnalyzeRequest(BaseModel):
    state: Dict = Field(..., description="pokeralpha.observed/v1 manual state")
    settings: AnalysisSettings = AnalysisSettings()
    include_receipt: bool = Field(
        True, description="attach a pokeralpha.receipt/v1 reproducibility "
                          "record to the response")


class ActionOut(BaseModel):
    label: str
    kind: str
    amount_to: Optional[float] = None
    frequency: Optional[float] = None       # solver mix / selection probability
    ev_bb: Optional[float] = None
    ev_se_bb: Optional[float] = None
    source: str = ""
    recommended: bool = False


class AbstentionOut(BaseModel):
    withheld: bool
    title: str
    reasons: List[str]
    reason_codes: List[str]
    fallback: Optional[str] = None


class RefusalOut(BaseModel):
    codes: List[str]
    message: str


class ProvenanceOut(BaseModel):
    input_source: str = "manual"
    gate_reasons: List[str] = []
    gate_signals: Dict[str, Optional[float]] = {}
    source_cascade: List[Dict] = []
    ev_note: Optional[str] = None           # provenance of EVs on solver mixes


class AnalyzeResponse(BaseModel):
    api: str = API_VERSION
    state_summary: str = ""
    actions: List[ActionOut] = []
    recommended: Optional[str] = None
    recommended_tag: str = "RECOMMENDED"    # HIGHEST FREQUENCY for solver mixes
    mix_meaning: str = ""
    method: str = ""
    source: str = "heuristic"               # solver | rollout | heuristic | none
    solver_state: str = "OFF"               # ACCEPTED | LOW CONFIDENCE | REJECTED | OFF
    decision_confidence: str = "low"
    ev_is_decision_basis: bool = False
    equity: Optional[float] = None
    equity_se: Optional[float] = None
    pot_bb: Optional[float] = None
    to_call_bb: Optional[float] = None
    pot_odds: Optional[float] = None
    effective_stack_bb: Optional[float] = None
    warnings: List[str] = []
    provenance: ProvenanceOut = ProvenanceOut()
    abstention: Optional[AbstentionOut] = None
    refusal: Optional[RefusalOut] = None
    #: pokeralpha.receipt/v1 — state fingerprint, artifact SHA, config,
    #: seeds, gate verdict and result; its receipt_id is a stable
    #: reproducibility ID (not a cryptographic proof).
    receipt: Optional[Dict] = None


class VerifyReceiptRequest(BaseModel):
    receipt: Dict


class VerifyReceiptResponse(BaseModel):
    api: str = API_VERSION
    reproduced: bool
    mismatches: List[str] = []


class WhatIfRequest(BaseModel):
    state: Dict
    variable: str
    values: List[str]
    settings: AnalysisSettings = AnalysisSettings()


class WhatIfVariant(BaseModel):
    value: str
    solver_state: str
    method: str
    confidence: str
    strategy: str
    actions: List[ActionOut] = []


class WhatIfResponse(BaseModel):
    api: str = API_VERSION
    variable: str
    variants: List[WhatIfVariant]
    rejected: Dict[str, str] = {}           # value -> why it was impossible


class SessionReviewRequest(BaseModel):
    hands: Dict = Field(..., description="pokeralpha.hand/v1 document")
    settings: AnalysisSettings = AnalysisSettings(
        equity_simulations=400, rollout_simulations=300)


class SessionReviewResponse(BaseModel):
    api: str = API_VERSION
    review: Dict                            # pokeralpha.session_review/v1


class MatrixCell(BaseModel):
    hand: str
    combo: str
    status: str
    dominant: Optional[str] = None
    dominant_freq: Optional[float] = None
    mix: List[List] = []                    # [label, probability]
    visits: Optional[float] = None
    reasons: List[str] = []
    notes: List[str] = []


class PreflopMatrixResponse(BaseModel):
    api: str = API_VERSION
    node: str
    nodes: List[str]
    cells: List[List[MatrixCell]]
    stats: Dict[str, int]
    note: str = ("Frequencies are live lookups against the promoted strategy "
                 "artifact; dimmed statuses mean the confidence gate rejects "
                 "the state and no numbers are shown.")


class ReleaseInfo(BaseModel):
    api: str = API_VERSION
    strategy_path: str
    strategy_sha256: str
    confidence_path: str
    iterations: int
    config_signature: str
    encoder: str
    disclaimer: str


class DemoSpot(BaseModel):
    name: str
    state: Dict


class HealthResponse(BaseModel):
    status: str = "ok"
    api: str = API_VERSION
    solver_loaded: bool = False
