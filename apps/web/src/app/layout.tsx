import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { TopNav } from "@/components/Nav";
import { SECONDARY_NAV, SITE_URL } from "@/lib/site";

const DESCRIPTION =
  "An open research and engineering project for game-theoretic decision-making under hidden " +
  "information: a custom Hold'em engine, CFR/MCCFR solvers, a bit-exact C++17 training backend, " +
  "exact reduced-game validation, a calibrated abstention layer and live browser demos.";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: { default: "PokerAlpha", template: "%s · PokerAlpha" },
  description: DESCRIPTION,
  openGraph: {
    title: "PokerAlpha",
    description: DESCRIPTION,
    type: "website",
    url: "/",
  },
  twitter: { card: "summary", title: "PokerAlpha", description: DESCRIPTION },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="wrap">
          <TopNav />
          {children}
          <footer className="site">
            <nav className="footer-nav" aria-label="More pages">
              <span className="label">More</span>
              {SECONDARY_NAV.map(([href, label]) => (
                <Link key={href} href={href}>
                  {label}
                </Link>
              ))}
            </nav>
            Analysis only. PokerAlpha never clicks, bets or controls a poker
            client. Real-time use belongs in private, play-money or test
            environments where assistance is permitted; everywhere else, use
            post-session review. Hands and screenshots stay on your machine.
          </footer>
        </div>
      </body>
    </html>
  );
}
