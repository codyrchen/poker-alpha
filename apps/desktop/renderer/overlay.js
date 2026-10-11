// Overlay renderer: compact strategy summary from the local engine.
// All semantics come from pokeralpha.api/v1 responses; this file renders.

/* global window, document */
const api = window.pokeralpha;
const $ = (id) => document.getElementById(id);

const COMPACT = [300, 190];
const EXPANDED = [330, 460];

let base = null;
let spots = [];
let expanded = false;
let current = null; // last AnalyzeResponse

function badge(el, text, cls) {
  el.textContent = text;
  el.className = "badge " + (cls || "");
}

function pct(f) {
  return f == null ? "—" : Math.round(f * 100) + "%";
}

function render(r, title) {
  current = r;
  const body = $("body");
  if (!r) {
    body.innerHTML =
      '<div class="panel"><b>Engine unavailable</b><div class="small muted">' +
      "Start it manually: <code>python -m poker_alpha.api</code>, then " +
      "relaunch. Details under Logs.</div></div>";
    return;
  }
  if (r.refusal) {
    body.innerHTML =
      `<div class="panel"><b>${r.refusal.message}</b>` +
      '<div class="small muted">No recommendation for a state PokerAlpha ' +
      "cannot trust.</div></div>";
    return;
  }
  const rows = (r.actions || [])
    .slice(0, expanded ? 8 : 3)
    .map(
      (a, i) =>
        `<div class="row ${i === 0 ? "top" : ""}">` +
        `<span class="name">${a.label}</span>` +
        `<span class="freq">${pct(a.frequency)}</span></div>`,
    )
    .join("");
  const solverCls =
    r.solver_state === "ACCEPTED"
      ? "good"
      : r.solver_state === "LOW CONFIDENCE"
        ? "warn"
        : "";
  let html = "";
  if (title) html += `<div class="small muted">${title}</div>`;
  if (r.abstention && r.abstention.withheld) {
    html +=
      `<div class="panel"><b>${r.abstention.title}</b>` +
      `<div class="small muted">${(r.abstention.reasons || [])[0] || ""}` +
      (r.abstention.fallback ? ` — fallback: ${r.abstention.fallback}` : "") +
      "</div></div>";
  }
  html += rows;
  html +=
    `<div class="row small" style="margin-top:0.3rem">` +
    `<span class="badge ${solverCls}">Solver: ${r.solver_state}</span>` +
    `<span class="badge">${(r.decision_confidence || "").toUpperCase()}</span></div>`;
  if (expanded && r.provenance && r.provenance.ev_note) {
    html += `<div class="small muted" style="margin-top:0.3rem">${r.provenance.ev_note}</div>`;
  }
  body.innerHTML = html;
}

async function analyzeState(state, title) {
  if (!base) return render(null);
  $("body").innerHTML = '<div class="muted small">Analyzing…</div>';
  try {
    const r = await fetch(`${base}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state }),
    });
    if (!r.ok) throw new Error(await r.text());
    render(await r.json(), title);
  } catch (e) {
    $("body").innerHTML =
      `<div class="panel"><b>Analysis failed</b>` +
      `<div class="small muted">${String(e).slice(0, 200)}</div></div>`;
  }
}

async function init() {
  base = await api.engineBase();
  if (!base) {
    badge($("conn"), "ENGINE OFFLINE", "warn");
    render(null);
    return;
  }
  try {
    const v = await (await fetch(`${base}/version`)).json();
    if (v.api !== "pokeralpha.api/v1") {
      badge($("conn"), "API INCOMPATIBLE", "warn");
      $("body").innerHTML =
        `<div class="panel"><b>Incompatible engine API (${v.api})</b>` +
        '<div class="small muted">Update PokerAlpha and relaunch.</div></div>';
      return;
    }
  } catch {
    badge($("conn"), "ENGINE OFFLINE", "warn");
    render(null);
    return;
  }
  badge($("conn"), "LOCAL", "good");
  spots = await (await fetch(`${base}/demo-spots`)).json();
  const sel = $("spot");
  sel.innerHTML = spots
    .map((s, i) => `<option value="${i}">${s.name}</option>`)
    .join("");
  sel.onchange = () => analyzeState(spots[sel.value].state, "demo");
  $("manual").value = JSON.stringify(spots[0].state);
  await analyzeState(spots[0].state, "demo");
}

$("mode").onclick = () => {
  expanded = !expanded;
  $("expanded").classList.toggle("open", expanded);
  api.setSize(...(expanded ? EXPANDED : COMPACT));
  if (current) render(current);
};
$("ct").onclick = async () => {
  const on = await api.toggleClickThrough();
  badge($("conn"), on ? "CLICK-THROUGH" : base ? "LOCAL" : "ENGINE OFFLINE",
        on ? "warn" : base ? "good" : "warn");
};
$("hide").onclick = () => api.hide();
$("quit").onclick = () => api.quit();
$("analyze").onclick = () => {
  try {
    analyzeState(JSON.parse($("manual").value), "manual");
  } catch {
    $("body").innerHTML =
      '<div class="panel"><b>Invalid JSON</b><div class="small muted">' +
      "Paste a pokeralpha.observed/v1 state.</div></div>";
  }
};
$("logsbtn").onclick = async () => {
  const el = $("logs");
  el.style.display = el.style.display === "none" ? "block" : "none";
  el.textContent = (await api.engineLogs()).join("\n");
};

init();
