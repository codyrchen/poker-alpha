"""Play Mode visual style — one restrained CSS block, injected once.

Design rules (docs/ui_product_spec.md): the NORMAL flow is quiet — open
regions on the app's own background, separated by hairlines, using
Streamlit's theme variables so it reads correctly in light and dark.
Bordered, self-contained dark panels are reserved for EXCEPTIONAL states
(abstention, waiting, errors), which therefore stand out. One accent
color; green only for accepted/high, amber for caution, red only for real
errors; no gradients, no glow.
"""

from __future__ import annotations

ACCENT = "#2a78d6"      # same blue as the repo's benchmark figures
GOOD = "#2e9e5b"
CAUTION = "#c2882a"
ERROR = "#c25047"

# Exceptional panels are self-contained (fixed dark), so they are readable
# on either Streamlit theme and visually interrupt the quiet flow.
PANEL_BG = "#15181d"
PANEL_BORDER = "#2a2f36"
PANEL_INK = "#e8eaed"
MUTED = "#8a8f96"       # readable on light and dark backgrounds
HAIRLINE = "rgba(128, 131, 138, 0.28)"
TRACK = "rgba(128, 131, 138, 0.22)"
CARD_BG = "#f4f2ec"

CSS = f"""
<style>
/* ---- quiet (normal) regions: theme-adaptive ---------------------------- */
.pa-hand {{
  padding: 0.4rem 0.2rem 0.9rem 0.2rem;
  margin-bottom: 0.4rem;
  border-bottom: 1px solid {HAIRLINE};
  color: var(--text-color, inherit);
}}
.pa-actions {{
  padding: 0.7rem 0.2rem 0.6rem 0.2rem;
  color: var(--text-color, inherit);
}}
.pa-footer {{
  padding: 0.55rem 0.2rem 0.4rem 0.2rem;
  border-top: 1px solid {HAIRLINE};
  color: var(--text-color, inherit);
}}
.pa-footer .pa-stat .v {{ font-size: 0.92rem; }}
.pa-muted {{ color: {MUTED}; font-size: 0.82rem; }}

.pa-cardrow {{ display: flex; gap: 0.45rem; align-items: center;
               flex-wrap: wrap; margin: 0.15rem 0 0.35rem 0; }}
.pa-card {{
  display: inline-block; min-width: 2.15rem; text-align: center;
  background: {CARD_BG}; color: #111; border-radius: 6px;
  border: 1px solid #c9c5ba;
  font: 600 1.25rem/1.9 "SF Pro Text", -apple-system, "Segoe UI", sans-serif;
  padding: 0.12rem 0.3rem;
}}
.pa-card.red {{ color: #b3262c; }}
.pa-card.small {{ min-width: 1.75rem; font-size: 1.02rem; line-height: 1.7; }}
.pa-sep {{ color: {MUTED}; margin: 0 0.4rem; }}
.pa-stats {{ display: flex; gap: 1.6rem; flex-wrap: wrap; margin-top: 0.55rem; }}
.pa-stat .v {{ font-size: 1.05rem; font-weight: 600;
               color: var(--text-color, inherit); }}
.pa-stat .k {{ font-size: 0.72rem; color: {MUTED};
               text-transform: uppercase; letter-spacing: 0.06em; }}

.pa-action {{ display: flex; align-items: center; gap: 0.9rem;
              padding: 0.5rem 0.65rem; border-radius: 8px;
              border: 1px solid transparent; margin-bottom: 0.3rem; }}
.pa-action.rec {{ border-color: {ACCENT}; background: rgba(42,120,214,0.10); }}
.pa-action .name {{ flex: 0 0 9.5rem; font-size: 1.05rem; font-weight: 600;
                    color: var(--text-color, inherit); }}
.pa-action.rec .name {{ font-size: 1.2rem; }}
.pa-action .bar {{ flex: 1 1 auto; height: 6px; background: {TRACK};
                   border-radius: 3px; overflow: hidden; min-width: 60px; }}
.pa-action .bar > div {{ height: 100%; background: {ACCENT}; }}
.pa-action .freq {{ flex: 0 0 3.2rem; text-align: right; font-weight: 600;
                    color: var(--text-color, inherit);
                    font-variant-numeric: tabular-nums; }}
.pa-action .ev {{ flex: 0 0 7.5rem; text-align: right; color: {MUTED};
                  font-size: 0.82rem; font-variant-numeric: tabular-nums; }}
.pa-rec-tag {{ color: {ACCENT}; font-size: 0.72rem; font-weight: 700;
               letter-spacing: 0.08em; }}

.pa-status {{ display: flex; gap: 1.6rem; flex-wrap: wrap;
              align-items: baseline; }}
.pa-badge {{ display: inline-block; padding: 0.1rem 0.55rem;
             border-radius: 5px; font-size: 0.8rem; font-weight: 700;
             letter-spacing: 0.05em; }}
.pa-badge.high {{ background: rgba(46,158,91,0.16); color: {GOOD}; }}
.pa-badge.medium {{ background: rgba(194,136,42,0.16); color: {CAUTION}; }}
.pa-badge.low {{ background: rgba(194,136,42,0.16); color: {CAUTION}; }}
.pa-badge.neutral {{ background: rgba(138,143,150,0.16); color: {MUTED}; }}
.pa-badge.error {{ background: rgba(194,80,71,0.16); color: {ERROR}; }}

@media (max-width: 720px) {{
  .pa-action .ev {{ display: none; }}   /* keep action + frequency primary */
  .pa-action .name {{ flex-basis: 8rem; }}
}}

/* ---- exceptional states: self-contained bordered panels ---------------- */
.pa-panel {{
  background: {PANEL_BG};
  border: 1px solid {PANEL_BORDER};
  border-radius: 10px;
  padding: 1.1rem 1.3rem;
  margin-bottom: 0.9rem;
  color: {PANEL_INK};
}}
.pa-panel .pa-stat .v {{ color: {PANEL_INK}; }}
.pa-abstain {{ border-left: 3px solid {CAUTION}; }}
.pa-abstain h4 {{ margin: 0 0 0.25rem 0; color: {CAUTION};
                  font-size: 0.85rem; letter-spacing: 0.08em; }}
.pa-abstain .title {{ font-size: 1.25rem; font-weight: 650; color: {PANEL_INK};
                      margin-bottom: 0.4rem; }}
.pa-abstain ul {{ margin: 0.3rem 0 0.3rem 1.1rem; padding: 0;
                  color: {PANEL_INK}; }}
.pa-wait {{ border-left: 3px solid {MUTED}; }}
.pa-wait .title {{ font-size: 1.15rem; font-weight: 600; color: {PANEL_INK}; }}
</style>
"""
