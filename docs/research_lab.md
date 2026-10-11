# The Research Lab (apps/web → /lab)

Interactive research surfaces, all driven by PRECOMPUTED real engine
output from one deterministic build:

    .venv/bin/python scripts/build_web_research_data.py

Every artifact is a `pokeralpha.web/v1` envelope carrying the generating
commit, release-strategy SHA-256, seed and arguments; every page ends
with a lab note (method / limitation / data / reproduce). React renders;
Python computes — policy numbers are never hand-written.

| route | question | data |
|---|---|---|
| /learn | can you watch strategy learning happen? | kuhn_reference.json + live browser CFR |
| /arena | can you exploit a trained strategy? | kuhn_reference.json (policy) + in-tab play |
| /surface | where does the decision change? | surface.json (81 real analyses) |
| /trust | where does the solver trust itself? | trust_map.json (169×3 with gate signals) |
| /microscope | how did one policy evolve across seeds? | microscope.json (9 artifacts) |
| /atlas | what kinds of policies were learned? | atlas.json (6k sampled infosets, SVD) |
| /break-it | where is it fragile? | break_it.json + redteam.json |
| /replay | what did the system believe in a hand? | one_hand.json (range stages, receipt) |
| /agents | who beats whom, with what uncertainty? | tournament.json |
| /games | does this extend beyond poker? | games.json (Kuhn/Leduc/auction) |
| /evidence | can every number be traced? | claims.json (registry) |

Public deployment note: everything above works without a local engine;
nothing live is faked, and PRECOMPUTED labeling is part of the page
contract (tested).
