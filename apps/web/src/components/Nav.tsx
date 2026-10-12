"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { PRIMARY_NAV, REPO_URL } from "@/lib/site";

export function TopNav() {
  const path = usePathname() ?? "/";
  const active = (href: string) => (href === "/" ? path === "/" : path === href || path.startsWith(href + "/"));
  return (
    <nav className="top" aria-label="Primary">
      <Link href="/" className="brand">
        PokerAlpha
      </Link>
      {PRIMARY_NAV.map(([href, label]) => (
        <Link key={href} href={href} className={active(href) ? "active" : undefined}
              aria-current={active(href) ? "page" : undefined}>
          {label}
        </Link>
      ))}
      <a href={REPO_URL} target="_blank" rel="noreferrer">
        GitHub ↗
      </a>
    </nav>
  );
}
