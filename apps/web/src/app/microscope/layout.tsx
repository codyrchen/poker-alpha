import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Solver microscope",
  description: "One information set across checkpoints and seeds.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
