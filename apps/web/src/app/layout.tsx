import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "PokerAlpha",
  description:
    "High-performance imperfect-information game solving: MCCFR in C++17, " +
    "exact validation, calibrated abstention. Local-first analysis tools.",
};

const NAV = [
  ["/analyze", "Analyze"],
  ["/study", "Study"],
  ["/strategy", "Strategy"],
  ["/trainer", "Trainer"],
  ["/research", "Research"],
  ["/benchmarks", "Benchmarks"],
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
