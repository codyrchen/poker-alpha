import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Analyzer",
  description: "Run a hand state through the PokerAlpha decision pipeline (local engine).",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
