// The three public demos featured on the homepage and the Demo page.
// Screenshots in public/screens are real captures of these pages
// (scripts/capture_site_screens.py).

export interface FeaturedDemo {
  href: string;
  tag: string;
  title: string;
  text: string;
  scope: string;
  img: string;
  alt: string;
  cta: string;
}

export const FEATURED: FeaturedDemo[] = [
  {
    href: "/learn",
    tag: "Live in your browser",
    title: "Watch a solver learn",
    text: "Run counterfactual regret minimization on Kuhn poker and watch exploitability fall as the strategy improves.",
    scope: "Real CFR computation, checked against the Python reference. Not an animation.",
    img: "/screens/learn-card.webp",
    alt: "The Watch a solver learn page after 10,000 CFR iterations, with the exploitability curve and live strategy table",
    cta: "Run the solver",
  },
  {
    href: "/arena",
    tag: "Live in your browser",
    title: "Play the Arena",
    text: "Play heads-up Kuhn poker against the trained CFR policy, then see the exact exploitability of how you played.",
    scope: "A small-game sandbox. It does not use the Hold'em release strategy.",
    img: "/screens/arena-card.webp",
    alt: "The Arena page with a Kuhn poker hand against the CFR policy",
    cta: "Play a hand",
  },
  {
    href: "/lab/phase2",
    tag: "Interactive results",
    title: "Explore Phase II results",
    text: "Abstraction refinement, the native speedup, a resolving time-budget slider, translation costs and the ideas that failed.",
    scope: "Precomputed exact results from reduced games, with their controls.",
    img: "/screens/phase2-card.webp",
    alt: "The Phase II results page with exploitability bars for abstraction refinement strategies",
    cta: "Open the results",
  },
];
