import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Arena",
  description: "Play heads-up Kuhn poker against a CFR policy, then see the exploitability of your own strategy.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
