// Hidden-Information Game Lab: the solver stack beyond poker.

import Link from "next/link";
import data from "@/data/lab/games.json";
import { MixBar, ProvenanceFooter, Question } from "@/components/Lab";

const env = data as any;
const G = env.payload;

const DESCRIPTIONS: Record<string, { title: string; text: string; teaches: string }> = {
  kuhn: {
    title: "Kuhn poker",
    text: "Three cards, one betting round — the smallest game that forces bluffing and mixed strategies.",
    teaches: "why equilibria randomize",
  },
  leduc: {
    title: "Leduc poker",
    text: "Six cards, two betting rounds, a public board card — a standard mid-size research benchmark.",
    teaches: "public information updating private beliefs",
  },
  auction_duel: {
    title: "Sealed-bid auction duel",
    text: "Each player privately values an item, then bids blind. Zero-sum duel form (payoff = your surplus minus theirs; ties award nobody) — deliberately NOT the classic general-sum first-price auction.",
    teaches: "hidden private values outside poker entirely",
  },
};

export default function GamesPage() {
  return (
    <main>
      <h1>Hidden-Information Game Lab</h1>
      <Question>
        Does this extend beyond poker? The solver stack runs on a generic
        extensive-form Game interface — every game below is implemented
        against it, solved by the SAME CFR implementation, and verified by
        exact exploitability.
      </Question>

      <table className="data">
        <thead>
          <tr>
            <th>game</th>
            <th>states</th>
            <th>infosets</th>
            <th>terminals</th>
            <th>CFR iterations</th>
            <th>exact exploitability</th>
            <th>train time</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(G).map(([name, g]: [string, any]) => (
            <tr key={name}>
              <td>{DESCRIPTIONS[name]?.title ?? name}</td>
              <td className="mono">{g.complexity.states.toLocaleString()}</td>
              <td className="mono">{g.complexity.infosets}</td>
              <td className="mono">{g.complexity.terminal_states.toLocaleString()}</td>
              <td className="mono">{g.cfr_iterations.toLocaleString()}</td>
              <td className="mono">{g.exploitability}</td>
              <td className="mono">{g.train_seconds}s</td>
            </tr>
          ))}
        </tbody>
      </table>

      {Object.entries(DESCRIPTIONS).map(([key, d]) => (
        <section key={key}>
          <h2>{d.title}</h2>
          <p style={{ maxWidth: "44rem" }}>
            {d.text} <span className="muted">Teaches: {d.teaches}.</span>
          </p>
          {key === "auction_duel" && G.auction_duel.strategy && (
            <>
              <p className="small muted">
                The CFR equilibrium (exploitability{" "}
                {G.auction_duel.exploitability}): bids shade BELOW value, and
                higher private values never bid less — classic
                hidden-information behavior, learned, not programmed:
              </p>
              <table className="data" style={{ maxWidth: "430px" }}>
                <thead>
                  <tr>
                    <th>private value</th>
                    <th>equilibrium bid mix</th>
                  </tr>
                </thead>
                <tbody>
                  {["0", "1", "2"].map((v) => (
                    <tr key={v}>
                      <td className="mono">value {v}</td>
                      <td>
                        <MixBar mix={G.auction_duel.strategy[`A${v}`]} />
                        <span className="small muted mono">
                          {Object.entries(
                            G.auction_duel.strategy[`A${v}`] as Record<string, number>,
                          )
                            .map(([b, p]) => `bid ${b}: ${(p * 100).toFixed(0)}%`)
                            .join(" · ")}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </section>
      ))}

      <h2>Poker is the flagship, not the limit</h2>
      <p style={{ maxWidth: "46rem" }}>
        PokerAlpha began with poker because it concentrates hidden
        information, bluffing, randomized strategies and adaptation into one
        measurable setting. The research infrastructure — the Game
        interface, the CFR family, exact evaluation, the agent tournament
        harness — applies to other SMALL imperfect-information games like
        the ones above. No claim is made about arbitrary real-world
        decisions.
      </p>
      <p className="small">
        Define a new game: subclass{" "}
        <span className="mono">poker_alpha.games.base.Game</span> (~60 lines
        for the auction duel, tests included) and every solver, evaluator
        and tournament tool works on it unchanged.{" "}
        <a href="https://github.com/codyrchen/poker-alpha/blob/main/poker_alpha/games/auction.py">
          Read the auction implementation
        </a>
      </p>
      <p>
        <Link className="btn" href="/learn">Watch CFR learn one of these live</Link>{" "}
        <Link className="btn" href="/agents">Agent tournaments</Link>
      </p>

      <ProvenanceFooter
        envelope={env}
        file="games.json"
        method="each game enumerated and solved by poker_alpha.solvers.cfr.CFRSolver; exploitability via exact best response on the full tree"
        limitation="small exactly solvable games only; the generic interface does not claim to scale every game to Hold'em size"
      />
    </main>
  );
}
