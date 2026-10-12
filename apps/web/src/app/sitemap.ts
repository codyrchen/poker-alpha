import type { MetadataRoute } from "next";
import { SITE_URL } from "@/lib/site";

export const dynamic = "force-static";

// Public, recruiter-facing routes first; deep-dive pages after.
const ROUTES = [
  "/", "/research", "/engineering", "/interactive", "/evidence",
  "/learn", "/arena", "/lab", "/lab/phase2", "/overview", "/tour", "/benchmarks",
  "/trust", "/surface", "/microscope", "/atlas", "/break-it", "/replay", "/agents",
  "/games", "/analyze", "/download",
];

export default function sitemap(): MetadataRoute.Sitemap {
  return ROUTES.map((r) => ({ url: `${SITE_URL}${r === "/" ? "" : r}`, changeFrequency: "monthly" }));
}
