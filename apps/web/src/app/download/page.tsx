import type { Metadata } from "next";
import Link from "next/link";
import { REPO_URL, RELEASES_URL } from "@/lib/site";
import { latestRelease, pickAssets, type Platform } from "@/lib/releases";

export const metadata: Metadata = {
  title: "Download",
  description: "Download PokerAlpha: desktop builds (planned), source code and local setup.",
};

const NAMES: Record<Platform, string> = { macos: "macOS", windows: "Windows", linux: "Linux" };
const doc = (p: string) => `${REPO_URL}/blob/main/${p}`;
const mb = (bytes?: number) => (bytes ? ` · ${Math.round(bytes / 1e6)} MB` : "");

export default async function DownloadPage() {
  const release = await latestRelease("codyrchen/poker-alpha");
  const downloads = pickAssets(release);
  const anyBuild = downloads.some((d) => d.asset);

  return (
    <main>
      <div className="eyebrow">Download</div>
      <h1>Download PokerAlpha</h1>
      <p className="lead" style={{ marginTop: "0.6rem" }}>
        PokerAlpha runs on your own machine. There is no hosted service and no account, and hands
        and screenshots stay local.
      </p>

      <section className="block" style={{ marginTop: "2.4rem" }}>
        <h2>Desktop app</h2>
        <p className="intro">
          {anyBuild
            ? `Builds from the ${release!.tag_name} release on GitHub.`
            : "Desktop builds are planned. When signed releases are published on GitHub, this page will link them here. Until then the app runs from source."}
        </p>
        <div className="platforms">
          {downloads.map(({ platform, asset }) => (
            <div key={platform}>
              <b>{NAMES[platform]}</b>
              {asset ? (
                <div className="state">
                  <a href={asset.browser_download_url}>{asset.name}</a>
                  {mb(asset.size)}
                </div>
              ) : (
                <div className="state">Coming soon</div>
              )}
            </div>
          ))}
        </div>
        {anyBuild && (
          <p className="small" style={{ marginTop: "0.8rem" }}>
            <a href={release!.html_url}>Release notes and checksums</a>
          </p>
        )}
      </section>

      <section className="block" style={{ marginTop: "3rem" }}>
        <h2>Source code</h2>
        <p className="intro">
          The engine, solvers, C++ backend, experiments, result files and this site are in one
          repository.
        </p>
        <div className="cta-row" style={{ marginTop: "0.9rem" }}>
          <a className="btn solid" href={REPO_URL}>View on GitHub</a>
          <a className="textlink" href={RELEASES_URL}>Releases</a>
        </div>
      </section>

      <section className="block" style={{ marginTop: "3rem" }}>
        <h2>Local development</h2>
        <p className="intro">Python 3.11+ and Node 20.9+. The C++ backend is optional and builds with pip.</p>
        <pre className="mono panel" style={{ whiteSpace: "pre-wrap", marginTop: "0.8rem" }}>
          {`git clone ${REPO_URL}
cd poker-alpha
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ui,api]"
pip install ./cpp                  # optional: native C++ training backend

python -m poker_alpha.api          # local engine on 127.0.0.1
cd apps/web && npm install && npm run dev   # this site, talking to the local engine`}
        </pre>
        <ul style={{ marginLeft: "1.2rem", marginTop: "0.8rem" }}>
          <li><a href={doc("README.md#quick-start")}>Quick start</a> and the full test suite</li>
          <li><a href={doc("docs/web_app.md")}>Web app and local engine</a></li>
          <li><a href={doc("docs/desktop_app.md")}>Desktop app</a>: build the Electron overlay with the engine bundled inside (unsigned)</li>
        </ul>
        <p className="small muted" style={{ marginTop: "0.8rem" }}>
          The live demos on <Link href="/interactive">the Demo page</Link> need none of this.
        </p>
      </section>

      <section className="block" style={{ marginTop: "3rem" }}>
        <h2>Intended use</h2>
        <ul style={{ marginLeft: "1.2rem" }}>
          <li>Analysis only. PokerAlpha never clicks, bets or controls a poker client.</li>
          <li>
            Real-time use only in private, play-money or test environments where assistance is
            permitted; everywhere else, use post-session review.
          </li>
          <li>
            The screen observer is experimental and excluded from the product. See the{" "}
            <a href={doc("docs/system_card.md")}>system card</a>.
          </li>
        </ul>
      </section>
    </main>
  );
}
