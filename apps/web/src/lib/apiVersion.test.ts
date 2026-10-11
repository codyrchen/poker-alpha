// Phase II reliability: API-version mismatch must never be rendered silently.
import { afterEach, describe, expect, it, vi } from "vitest";
import { API_VERSION, ApiVersionError, analyze, checkHealth, preflopMatrix, requireApiVersion } from "./api";

function mockFetch(body: unknown, ok = true) {
  return vi.fn(async () => ({ ok, status: ok ? 200 : 500, json: async () => body, text: async () => "err" })) as any;
}

afterEach(() => vi.unstubAllGlobals());

describe("API version contract", () => {
  it("accepts matching versions and rejects anything else", () => {
    expect(requireApiVersion({ api: API_VERSION })).toEqual({ api: API_VERSION });
    expect(() => requireApiVersion({ api: "pokeralpha.api/v2" })).toThrow(ApiVersionError);
    expect(() => requireApiVersion({} as any)).toThrow(ApiVersionError);
  });

  it("health reports a mismatched engine as not connected", async () => {
    vi.stubGlobal("fetch", mockFetch({ status: "ok", api: API_VERSION }));
    expect(await checkHealth("http://127.0.0.1:8765")).toBe(true);
    vi.stubGlobal("fetch", mockFetch({ status: "ok", api: "pokeralpha.api/v0" }));
    expect(await checkHealth("http://127.0.0.1:8765")).toBe(false);
  });

  it("data calls throw on a version mismatch instead of returning data", async () => {
    vi.stubGlobal("fetch", mockFetch({ api: "pokeralpha.api/v2", actions: [] }));
    await expect(analyze("http://127.0.0.1:8765", {})).rejects.toBeInstanceOf(ApiVersionError);
    await expect(preflopMatrix("http://127.0.0.1:8765", "open")).rejects.toBeInstanceOf(ApiVersionError);
  });
});
