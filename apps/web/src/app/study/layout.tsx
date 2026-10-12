import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Session review",
  description: "Post-session review of a hand history (local engine).",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
