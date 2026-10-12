import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Strategy explorer",
  description: "Browse the release strategy's preflop matrix.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
