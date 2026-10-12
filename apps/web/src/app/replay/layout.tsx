import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Hand replay",
  description: "One decision with the modelled opponent range street by street.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
