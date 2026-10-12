import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Decision surface",
  description: "Where the recommendation changes over bet size and stack depth.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
