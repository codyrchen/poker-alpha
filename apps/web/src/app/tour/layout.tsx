import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Tour",
  description: "PokerAlpha in ten short steps.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
