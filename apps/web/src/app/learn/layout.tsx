import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Watch a solver learn",
  description: "Vanilla CFR on Kuhn poker running live in the browser, with exact exploitability.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
