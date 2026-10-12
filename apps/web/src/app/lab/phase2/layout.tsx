import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Phase II results",
  description: "Controlled experiments: abstraction refinement, native speedup, resolving, translation and negative results.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
