// Desktop release detection for the download page.
//
// The site is a static export, so this runs once at build time. It looks at
// the latest published GitHub release and links an asset for a platform only
// when one with a matching installer extension exists. Anything else (no
// release, a draft, no network, rate limiting) leaves that platform as
// "coming soon". Set SKIP_RELEASE_CHECK=1 to build fully offline.

export type Platform = "macos" | "windows" | "linux";

export interface ReleaseAsset {
  name: string;
  browser_download_url: string;
  size?: number;
}

export interface Release {
  tag_name: string;
  html_url: string;
  draft?: boolean;
  prerelease?: boolean;
  assets: ReleaseAsset[];
}

export interface PlatformDownload {
  platform: Platform;
  asset: ReleaseAsset | null;
}

const EXTENSIONS: Record<Platform, string[]> = {
  macos: [".dmg", "-mac.zip", "-macos.zip", "-darwin.zip"],
  windows: [".exe", ".msi"],
  linux: [".appimage", ".deb", ".rpm"],
};

export const PLATFORMS: Platform[] = ["macos", "windows", "linux"];

export function pickAssets(release: Release | null): PlatformDownload[] {
  const usable = release && !release.draft ? release.assets : [];
  return PLATFORMS.map((platform) => ({
    platform,
    asset:
      usable.find((a) => {
        const n = a.name.toLowerCase();
        return EXTENSIONS[platform].some((ext) => n.endsWith(ext));
      }) ?? null,
  }));
}

export async function latestRelease(repo: string): Promise<Release | null> {
  if (process.env.SKIP_RELEASE_CHECK === "1") return null;
  try {
    const res = await fetch(`https://api.github.com/repos/${repo}/releases/latest`, {
      headers: { Accept: "application/vnd.github+json" },
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return null; // 404 = no published release yet
    const data = (await res.json()) as Release;
    return Array.isArray(data.assets) ? data : null;
  } catch {
    return null;
  }
}
