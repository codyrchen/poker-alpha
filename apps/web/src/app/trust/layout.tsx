import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Trust map",
  description: "Preflop gate verdicts, visits and cross-seed disagreement.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
