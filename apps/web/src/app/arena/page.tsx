"use client";

// PokerAlpha Arena: play Kuhn poker against the CFR-trained policy,
// entirely in this tab. No accounts, no currency, no uploads — chips are
// game units only. After the session: X-ray replay, exact
// exploitability of YOUR empirical strategy, and a sample-size-aware
// strategy fingerprint.

import Link from "next/link";
import { useRef, useState } from "react";
import ref from "@/data/lab/kuhn_reference.json";
import { MixBar, Question } from "@/components/Lab";
import { ArenaMatch, fingerprint, xray } from "@/lib/arena";
import { CARD_NAMES } from "@/lib/kuhn";

const policy = (ref as any).payload.final_policy;

const HIST_TEXT: Record<string, string> = {
  "": "You act first.",
  p: "PokerAlpha checked.",
  b: "PokerAlpha bet 1 chip.",
  pb: "You checked; PokerAlpha bet 1 chip.",
};

function describeEnd(history: string, won: boolean, aiCard: number): string {
  const showdown = ["pp", "bb", "pbb"].includes(history);
  const base = showdown
    ? `Showdown — PokerAlpha had the ${CARD_NAMES[aiCard]}.`
    : history.endsWith("p") && history.includes("b")
      ? history === "bp" || history === "pbp"
        ? "Fold."
        : "Fold."
      : "";
  return `${base} ${won ? "You win" : "You lose"} the pot.`;
}

export default function ArenaPage() {
  const matchRef = useRef<ArenaMatch | null>(null);
  const [, bump] = useState(0);
  const [target, setTarget] = useState(10);
  const [phase, setPhase] = useState<"intro" | "play" | "report">("intro");
  const shareRef = useRef<HTMLCanvasElement>(null);

  const match = matchRef.current;

  function start(hands: number) {
    matchRef.current = new ArenaMatch(policy, Date.now() & 0x7fffffff);
    setTarget(hands);
    matchRef.current.startHand();
    setPhase("play");
    bump((v) => v + 1);
  }

  function act(a: "p" | "b") {
    match!.humanAct(a);
    if (match!.toAct === "done") {
      if (match!.session.hands.length >= target) setPhase("report");
    }
    bump((v) => v + 1);
  }

  function nextHand() {
    match!.startHand();
    bump((v) => v + 1);
  }

  function downloadCard() {
    const x = xray(match!.session);
    const fp = fingerprint(match!.session);
    const c = shareRef.current!;
    c.width = 720;
    c.height = 420;
    const g = c.getContext("2d")!;
    g.fillStyle = "#0e1013";
    g.fillRect(0, 0, 720, 420);
    g.fillStyle = "#e8eaed";
    g.font = "700 26px -apple-system, sans-serif";
    g.fillText("PokerAlpha — Strategy Fingerprint", 32, 52);
    g.fillStyle = "#8a8f96";
    g.font = "14px -apple-system, sans-serif";
    g.fillText(
      `Kuhn Arena · ${x.hands} hands · ${x.decisions} decisions · observed tendencies in this sample only`,
      32,
      80,
    );
    let y = 130;
    for (const f of fp) {
      g.fillStyle = "#e8eaed";
      g.font = "15px -apple-system, sans-serif";
      g.fillText(f.label, 32, y);
      g.fillStyle = "#8a8f96";
      const val =
        f.value === null
          ? `insufficient data (n=${f.n})`
          : f.label.includes("surprise")
            ? `${f.value.toFixed(2)} bits (n=${f.n})`
            : `${(f.value * 100).toFixed(0)}% ± ${(1.96 * (f.se ?? 0) * 100).toFixed(0)} (n=${f.n})`;
      g.fillText(val, 430, y);
      if (f.value !== null && !f.label.includes("surprise")) {
        g.fillStyle = "#2a3550";
        g.fillRect(32, y + 8, 370, 7);
        g.fillStyle = "#2a78d6";
        g.fillRect(32, y + 8, 370 * f.value, 7);
      }
      y += 48;
    }
    if (x.exploitability !== null) {
      g.fillStyle = "#c2882a";
      g.font = "15px -apple-system, sans-serif";
      g.fillText(
        `Exact exploitability of your observed strategy: ${x.exploitability.toFixed(3)} chips/hand`,
        32,
        y + 6,
      );
    }
    g.fillStyle = "#5f6368";
    g.font = "12px -apple-system, sans-serif";
    g.fillText("pokeralpha · local-only · no accounts · github.com/codyrchen/poker-alpha", 32, 396);
    const a = document.createElement("a");
    a.download = "pokeralpha-fingerprint.png";
    a.href = c.toDataURL("image/png");
    a.click();
  }

  if (phase === "intro") {
    return (
      <main>
        <h1>Arena</h1>
        <Question>
          Can you find a weakness in a trained strategy? Play heads-up Kuhn
          poker against PokerAlpha&rsquo;s CFR policy — then see exactly what
          it observed about you.
        </Question>
        <h2>Kuhn poker in 15 seconds</h2>
        <p style={{ maxWidth: "44rem" }}>
          Three cards: <b>J &lt; Q &lt; K</b>. You each ante 1 chip and get
          one private card. One round: <b>check/fold</b> (&ldquo;pass&rdquo;)
          or <b>bet/call</b> one chip. If a bet is called, the higher card
          wins; fold and the bettor takes the pot. Tiny game — but it already
          forces bluffing and mixed strategies, which is why it is the
          standard first test of a poker solver.
        </p>
        <p className="muted small">
          The opponent is the committed CFR policy from the{" "}
          <Link href="/learn">learning lab</Link> (exploitability 0.0015
          chips/hand). Everything runs and stays in this browser tab. This is
          a research sandbox owned by PokerAlpha — not a gambling product and
          not connected to any poker site.
        </p>
        <div style={{ display: "flex", gap: "0.6rem", margin: "1.4rem 0" }}>
          <button onClick={() => start(10)}>Play 10 hands</button>
          <button className="primary" onClick={() => start(25)}>
            Play 25 hands
          </button>
          <button onClick={() => start(50)}>Play 50 hands</button>
        </div>
      </main>
    );
  }

  const h = match!.current!;
  const myCard = h.cards[h.humanSeat];
  const aiCard = h.cards[1 - h.humanSeat];
  const done = match!.toAct === "done";
  const facingBet = h.history.endsWith("b");

  if (phase === "play") {
    return (
      <main>
        <h1>Arena</h1>
        <p className="muted small">
          Hand {h.index + 1} of {target} · you are{" "}
          {h.humanSeat === 0 ? "first" : "second"} to act this hand · score:{" "}
          <span className="mono">
            {match!.session.chips >= 0 ? "+" : ""}
            {match!.session.chips} chips
          </span>
        </p>
        <div className="arena-table">
          <div className="muted small">PokerAlpha</div>
          <div className={`arena-card ${done ? "" : "back"}`}>
            {done ? CARD_NAMES[aiCard] : "?"}
          </div>
          <div style={{ margin: "1rem 0" }} className="muted">
            {done
              ? describeEnd(h.history, h.utilityToHuman > 0, aiCard)
              : HIST_TEXT[h.history.slice(h.humanSeat === 0 ? 0 : 1)] ??
                HIST_TEXT[h.history] ??
                ""}
          </div>
          <div className="arena-card">{CARD_NAMES[myCard]}</div>
          <div className="muted small">you</div>
          {!done ? (
            <div className="arena-actions">
              <button onClick={() => act("p")}>
                {facingBet ? "Fold" : "Check"}
              </button>
              <button className="primary" onClick={() => act("b")}>
                {facingBet ? "Call" : "Bet 1"}
              </button>
            </div>
          ) : (
            <div className="arena-actions">
              <span
                className={`badge ${h.utilityToHuman > 0 ? "good" : "warn"}`}
                style={{ alignSelf: "center" }}
              >
                {h.utilityToHuman > 0 ? "+" : ""}
                {h.utilityToHuman} chips
              </span>
              {match!.session.hands.length < target ? (
                <button className="primary" onClick={nextHand}>
                  Next hand
                </button>
              ) : (
                <button className="primary" onClick={() => setPhase("report")}>
                  See the X-ray report
                </button>
              )}
            </div>
          )}
        </div>
        <p className="muted small">
          During play PokerAlpha never shows its card or its mix — that would
          invalidate the game. Everything is revealed in the report.
        </p>
      </main>
    );
  }

  // report
  const x = xray(match!.session);
  const fp = fingerprint(match!.session);
  return (
    <main>
      <h1>X-ray report</h1>
      <p className="lead">{x.verdict}</p>

      <h2>Strategy fingerprint</h2>
      <p className="muted small">
        Observed tendencies in THIS sample — behavior, not personality.
        Dimensions without enough observations say so.
      </p>
      <table className="data" style={{ maxWidth: "680px" }}>
        <tbody>
          {fp.map((f) => (
            <tr key={f.label}>
              <td>
                {f.label}
                <div className="muted small">{f.note}</div>
              </td>
              <td className="mono" style={{ whiteSpace: "nowrap" }}>
                {f.value === null
                  ? `insufficient (n=${f.n})`
                  : f.label.includes("surprise")
                    ? `${f.value.toFixed(2)} bits`
                    : `${(f.value * 100).toFixed(0)}% ±${(1.96 * (f.se ?? 0) * 100).toFixed(0)} (n=${f.n})`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {x.exploitability !== null && (
        <>
          <h2>How exploitable were you?</h2>
          <p style={{ maxWidth: "44rem" }}>
            Treating your observed frequencies as your strategy (
            {x.coverage.observed}/{x.coverage.total} situations observed;
            unobserved ones assumed uniform), an exactly best-responding
            opponent would win{" "}
            <b className="mono">{x.exploitability.toFixed(3)} chips/hand</b>{" "}
            against it. The trained policy&rsquo;s own number is 0.0015. This
            is an exact tree computation, but your &ldquo;strategy&rdquo; here
            is a small-sample estimate — play more hands and it changes.
          </p>
        </>
      )}

      <h2>Every decision, against the trained mix</h2>
      <table className="data" style={{ maxWidth: "680px" }}>
        <thead>
          <tr>
            <th>hand</th>
            <th>your card / spot</th>
            <th>you chose</th>
            <th style={{ width: "30%" }}>trained mix (p / b)</th>
          </tr>
        </thead>
        <tbody>
          {match!.session.hands.flatMap((hand) =>
            hand.humanDecisions.map((d, i) => (
              <tr key={`${hand.index}-${i}`}>
                <td className="mono">{hand.index + 1}</td>
                <td>
                  {CARD_NAMES[Number(d.key[0])]}
                  {d.key.length > 1 ? ` after "${d.key.slice(1)}"` : ", first to act"}
                </td>
                <td className="mono">{d.action === "p" ? "pass" : "bet/call"}</td>
                <td>
                  <MixBar mix={d.aiMix} />
                </td>
              </tr>
            )),
          )}
        </tbody>
      </table>

      <div style={{ display: "flex", gap: "0.6rem", margin: "1.6rem 0", flexWrap: "wrap" }}>
        <button className="primary" onClick={downloadCard}>
          Download fingerprint card (PNG)
        </button>
        <button onClick={() => setPhase("intro")}>Play again</button>
        <Link className="btn" href="/learn">
          How the opponent was trained
        </Link>
      </div>
      <canvas ref={shareRef} style={{ display: "none" }} />
      <p className="muted small">
        The card is rendered locally in your browser; nothing was uploaded
        anywhere. No ratings, no leaderboards — {x.hands} hands is a sample,
        not a skill certificate.
      </p>
    </main>
  );
}
