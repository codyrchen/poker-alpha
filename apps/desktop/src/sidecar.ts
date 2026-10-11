// Python engine sidecar: launch `python -m poker_alpha.api --port 0`, read
// the announced port, health-check it, and shut the owned process down
// cleanly. If an engine is already running on the default port we attach to
// it instead of launching a duplicate.

import { ChildProcess, spawn } from "child_process";
import { existsSync } from "fs";
import * as path from "path";

export interface EngineHandle {
  base: string; // http://127.0.0.1:<port>
  owned: ChildProcess | null; // null when we attached to an existing engine
}

const DEFAULT_BASE = "http://127.0.0.1:8765";

export function repoRoot(): string {
  // apps/desktop/dist -> repo root
  return path.resolve(__dirname, "..", "..", "..");
}

export function pythonPath(): string {
  if (process.env.POKERALPHA_PYTHON) return process.env.POKERALPHA_PYTHON;
  const venv = path.join(repoRoot(), ".venv", "bin", "python");
  if (existsSync(venv)) return venv;
  return "python3";
}

export async function health(base: string, timeoutMs = 1500): Promise<boolean> {
  try {
    const r = await fetch(`${base}/health`, {
      signal: AbortSignal.timeout(timeoutMs),
    });
    const j: any = await r.json();
    return j.status === "ok" && j.api === "pokeralpha.api/v1";
  } catch {
    return false;
  }
}

async function wait(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

export async function startEngine(
  log: (line: string) => void,
): Promise<EngineHandle> {
  // 1) attach to an already-running engine (prevents duplicate launches)
  if (await health(DEFAULT_BASE)) {
    log(`attached to existing engine at ${DEFAULT_BASE}`);
    return { base: DEFAULT_BASE, owned: null };
  }
  // 2) launch our own on a free port (127.0.0.1 only — the API refuses
  //    non-loopback hosts by design). POKERALPHA_ENGINE_BIN points at a
  //    frozen engine binary (packaging/dist/pokeralpha-engine); otherwise
  //    the repo's Python environment runs the module directly.
  const frozen = process.env.POKERALPHA_ENGINE_BIN;
  let child: ChildProcess;
  let launcher: string;
  if (frozen && existsSync(frozen)) {
    launcher = frozen;
    log(`launching frozen engine: ${frozen} --port 0`);
    child = spawn(frozen, ["--port", "0"], {
      stdio: ["ignore", "pipe", "pipe"],
    });
  } else {
    const py = pythonPath();
    launcher = py;
    log(`launching sidecar: ${py} -m poker_alpha.api --port 0`);
    child = spawn(py, ["-m", "poker_alpha.api", "--port", "0"], {
      cwd: repoRoot(),
      stdio: ["ignore", "pipe", "pipe"],
    });
  }
  let port: number | null = null;
  child.stdout?.on("data", (b: Buffer) => {
    const m = /POKERALPHA_API_PORT=(\d+)/.exec(b.toString());
    if (m) port = Number(m[1]);
    log(b.toString().trim());
  });
  child.stderr?.on("data", (b: Buffer) => log(b.toString().trim()));

  const t0 = Date.now();
  while (Date.now() - t0 < 30000) {
    if (child.exitCode !== null)
      throw new Error(`engine exited with code ${child.exitCode} — is the ` +
        `Python environment installed? (looked for ${launcher})`);
    if (port !== null) {
      const base = `http://127.0.0.1:${port}`;
      if (await health(base, 1000)) {
        log(`engine healthy at ${base}`);
        return { base, owned: child };
      }
    }
    await wait(250);
  }
  child.kill();
  throw new Error("engine did not become healthy within 30s");
}

export function stopEngine(h: EngineHandle, log: (l: string) => void): void {
  if (h.owned && h.owned.exitCode === null) {
    log("stopping owned engine");
    h.owned.kill("SIGTERM");
    // last resort after 3s
    const owned = h.owned;
    setTimeout(() => {
      if (owned.exitCode === null) owned.kill("SIGKILL");
    }, 3000).unref();
  }
}
