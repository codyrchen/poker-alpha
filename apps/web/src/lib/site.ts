// Site-wide configuration: navigation, canonical URL and author details.
// Everything recruiter-facing that is not derived from research data lives
// here, so it can be checked and edited in one place.

export const REPO_URL = "https://github.com/codyrchen/poker-alpha";

// TODO(deploy): set SITE_URL (e.g. in the Vercel project settings) to the
// public origin. It is used for canonical metadata, OpenGraph and the sitemap.
export const SITE_URL = (process.env.SITE_URL ?? "http://localhost:3000").replace(/\/$/, "");

export const PRIMARY_NAV: readonly [string, string][] = [
  ["/", "Overview"],
  ["/research", "Research"],
  ["/engineering", "Engineering"],
  ["/interactive", "Interactive"],
  ["/evidence", "Evidence"],
];

// Deep-dive pages that stay reachable from the footer and hub pages.
export const SECONDARY_NAV: readonly [string, string][] = [
  ["/overview", "One-page overview"],
  ["/tour", "Tour"],
  ["/lab", "Research lab"],
  ["/lab/phase2", "Phase II results"],
  ["/analyze", "Analyzer"],
  ["/download", "Run locally"],
];

export interface AuthorLink {
  label: string;
  href: string | null; // null = not provided yet (rendered as a TODO, never invented)
}

// Author details as given in the site brief.
// TODO(author): the repository owner and earlier site copy credit "Cody Chen"
// (github.com/codyrchen). Confirm the intended author before publishing.
export const AUTHOR = {
  name: "Krish Patel",
  school: "Carnegie Mellon University",
  program: "Statistics & Machine Learning + Artificial Intelligence",
  interests: "Interested in quantitative research, high-performance systems, and machine learning.",
  links: [
    { label: "GitHub", href: REPO_URL },
    { label: "LinkedIn", href: null }, // TODO(author): LinkedIn URL
    { label: "Resume", href: null }, // TODO(author): resume URL or /resume.pdf in public/
    { label: "Email", href: null }, // TODO(author): contact address
  ] as AuthorLink[],
};
