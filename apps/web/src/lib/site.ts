// Site-wide configuration: navigation, canonical URL and author details.
// Everything public-facing that is not derived from research data lives
// here, so it can be checked and edited in one place.

export const REPO_URL = "https://github.com/codyrchen/poker-alpha";
export const RELEASES_URL = `${REPO_URL}/releases`;

// Set SITE_URL (e.g. in the Vercel project settings) to the public origin.
// It is used for canonical metadata, OpenGraph and the sitemap.
export const SITE_URL = (process.env.SITE_URL ?? "http://localhost:3000").replace(/\/$/, "");

export const PRIMARY_NAV: readonly [string, string][] = [
  ["/", "Overview"],
  ["/interactive", "Demo"],
  ["/research", "Research"],
  ["/engineering", "Engineering"],
  ["/download", "Download"],
];

// Deep-dive pages that stay reachable from the footer and hub pages.
export const SECONDARY_NAV: readonly [string, string][] = [
  ["/evidence", "Evidence"],
  ["/lab/phase2", "Phase II results"],
  ["/lab", "Research lab"],
  ["/overview", "One-page overview"],
  ["/tour", "Tour"],
  ["/analyze", "Analyzer"],
];

export interface AuthorLink {
  label: string;
  href: string;
}

// Only links that exist are listed; nothing here is a placeholder.
export const AUTHOR = {
  name: "Cody Chen",
  school: "University of Mississippi",
  program: "Mathematics + Computer Science",
  interests:
    "Interested in quantitative research, high-performance systems, machine learning, and game-theoretic systems.",
  links: [
    { label: "GitHub", href: "https://github.com/codyrchen" },
    { label: "Source code", href: REPO_URL },
  ] as AuthorLink[],
};
