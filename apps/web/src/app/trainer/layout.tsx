import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Trainer",
  description: "Answer bundled spots and compare with the engine's mix.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
