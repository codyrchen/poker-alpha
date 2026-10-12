import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Policy atlas",
  description: "Stored policies projected to two dimensions.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
