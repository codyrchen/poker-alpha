import { describe, expect, it } from "vitest";
import { pickAssets, type Release } from "./releases";

const rel = (names: string[], draft = false): Release => ({
  tag_name: "v0.1.0",
  html_url: "https://github.com/x/y/releases/tag/v0.1.0",
  draft,
  assets: names.map((name) => ({ name, browser_download_url: `https://example.invalid/${name}` })),
});

describe("pickAssets", () => {
  it("reports every platform as unavailable without a release", () => {
    expect(pickAssets(null).every((p) => p.asset === null)).toBe(true);
  });

  it("matches installers by extension, case-insensitively", () => {
    const got = pickAssets(rel(["PokerAlpha-0.1.0.DMG", "PokerAlpha Setup 0.1.0.exe", "PokerAlpha-0.1.0.AppImage"]));
    expect(got.map((p) => p.asset?.name)).toEqual([
      "PokerAlpha-0.1.0.DMG",
      "PokerAlpha Setup 0.1.0.exe",
      "PokerAlpha-0.1.0.AppImage",
    ]);
  });

  it("ignores source archives and checksums", () => {
    const got = pickAssets(rel(["source.zip", "source.tar.gz", "SHA256SUMS.txt"]));
    expect(got.every((p) => p.asset === null)).toBe(true);
  });

  it("ignores draft releases", () => {
    expect(pickAssets(rel(["PokerAlpha.dmg"], true))[0].asset).toBeNull();
  });
});
