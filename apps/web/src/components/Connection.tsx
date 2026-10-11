"use client";

import { useEffect, useState } from "react";
import { apiBase, checkHealth, ConnState } from "@/lib/api";

export function useConnection(): { state: ConnState; base: string } {
  const [state, setState] = useState<ConnState>("checking");
  const base = apiBase();
  useEffect(() => {
    let live = true;
    const probe = async () => {
      const ok = await checkHealth(base);
      if (live) setState(ok ? "connected" : "offline");
    };
    probe();
    const t = setInterval(probe, 5000);
    return () => {
      live = false;
      clearInterval(t);
    };
  }, [base]);
  return { state, base };
}

export function ConnectionBadge({ state }: { state: ConnState }) {
  if (state === "checking") return <span className="badge mute">CHECKING ENGINE…</span>;
  if (state === "connected") return <span className="badge good">LOCAL ENGINE CONNECTED</span>;
  return (
    <span className="badge warn" title="Start it with: python -m poker_alpha.api">
      ENGINE OFFLINE — SHOWING PRE-COMPUTED OUTPUT
    </span>
  );
}

export function OfflineNote({ state }: { state: ConnState }) {
  if (state !== "offline") return null;
  return (
    <p className="muted small">
      No local engine is running, so the results below are <b>pre-computed
      outputs of the real engine</b> bundled with this page — not live
      computation. Start the engine with{" "}
      <code className="mono">python -m poker_alpha.api</code> to analyze your
      own states.
    </p>
  );
}
