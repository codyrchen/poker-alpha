import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "PokerAlpha — an interactive laboratory for decision-making under hidden information",
  description:
    "Custom Hold'em engine · game-theoretic solvers · 74× C++ training " +
    "backend (bit-exact) · exact-game validation · calibrated abstention · " +
    "play against it in the browser. Independent research project by Cody Chen.",
  openGraph: {
    title: "PokerAlpha",
    description:
      "Game-theoretic solver · 74× native acceleration · calibrated " +
      "uncertainty · watch it learn, play against it, break it.",
    type: "website",
  },
};

const NAV = [
  ["/overview", "Overview"],
  ["/tour", "Tour"],
  ["/lab", "Research Lab"],
  ["/arena", "Arena"],
  ["/engineering", "Engineering"],
  ["/analyze", "Analyze"],
  ["/download", "Download"],
] as const;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="wrap">
          <nav className="top">
            <Link href="/" className="brand">
              PokerAlpha
            </Link>
            {NAV.map(([href, label]) => (
              <Link key={href} href={href}>
                {label}
              </Link>
            ))}
            <a
              href="https://github.com/codyrchen/poker-alpha"
              target="_blank"
              rel="noreferrer"
            >
              GitHub ↗
            </a>
          </nav>
          {children}
          <footer className="site">
            Analysis only — PokerAlpha never clicks, bets or controls a poker
            client. Real-time use belongs in private, play-money or test
            environments where assistance is permitted; everywhere else, use
            post-session review. Hands and screenshots stay on your machine.
          </footer>
        </div>
      </body>
    </html>
  );
}
