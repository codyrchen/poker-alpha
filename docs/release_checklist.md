# Release checklist (product surfaces)

## Every release

- [ ] `pytest` green (fast + `-m slow`), CI green on main matrix
- [ ] `apps/web`: `npm run test && npm run typecheck && npm run build`
- [ ] `apps/desktop`: `npm run typecheck && npm run smoke`
- [ ] demos run: `python -m poker_alpha.demo | holdem_demo | platform_demo`
- [ ] API boot smoke: `python -m poker_alpha.api --port 0` → `/health`
- [ ] Streamlit boot smoke (Play, Study, Developer all render)
- [ ] `git status` clean; no large accidental files
      (`git diff --stat`, check anything > 1 MB)
- [ ] no credentials / private captures committed
      (`tests/test_privacy.py` + `.gitignore` audit)

## Strategy / solver changes only

- [ ] artifact integrity tests (`test_artifact_integrity.py`,
      `test_release_strategy.py`)
- [ ] confidence table SHA-bound to the artifact
- [ ] docs/solver_status_final.md updated with evidence, not adjectives
- [ ] README claims re-checked against committed results

## Claims hygiene (always)

- [ ] no GTO claims; "experimental abstract strategy" wording intact
- [ ] observer: REAL HELD-OUT VALIDATION remains BLOCKED until
      independent sessions are evaluated; no accuracy numbers without
      `poker_alpha.observer.dataset baseline` output to cite
- [ ] regret language: "ESTIMATED DECISION REGRET", never "money lost"
- [ ] packaging: "unsigned local development preview" until
      signing/notarization exists

## External blockers (not release blockers for the local product)

signed macOS/Windows builds (credentials) · public hosting (deployment
credentials/domain) · observer accuracy claims (independent labeled
sessions) · ML model quality claims (enough real data).
