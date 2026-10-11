// PokerAlpha desktop overlay — local development preview.
//
// A transparent, draggable, always-on-top window showing the engine's
// compact strategy summary (the compact_summary semantics), with an
// expanded state for the full mix and a hidden state. Analysis only: there
// is no clicking, betting, client control or stealth of any kind — the
// window is a normal, visible, user-controlled overlay.

import { app, BrowserWindow, globalShortcut, ipcMain } from "electron";
import * as path from "path";
import { EngineHandle, startEngine, stopEngine } from "./sidecar";

const SMOKE = process.argv.includes("--smoke");

let win: BrowserWindow | null = null;
let engine: EngineHandle | null = null;
let clickThrough = false;
const logs: string[] = [];

function log(line: string): void {
  if (!line) return;
  logs.push(`[${new Date().toISOString()}] ${line}`);
  if (logs.length > 500) logs.shift();
  if (SMOKE) console.log(line);
}

function createWindow(): void {
  win = new BrowserWindow({
    width: 300,
    height: 190,
    frame: false,
    transparent: true,
    resizable: true,
    alwaysOnTop: true,
    hasShadow: false,
    skipTaskbar: false, // visible in the Dock/taskbar on purpose: no stealth
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.setAlwaysOnTop(true, "floating");
  win.loadFile(path.join(__dirname, "..", "renderer", "overlay.html"));
}

ipcMain.handle("engine-base", () => engine?.base ?? null);
ipcMain.handle("engine-logs", () => logs.slice(-80));
ipcMain.handle("set-size", (_e, w: number, h: number) => {
  win?.setSize(Math.round(w), Math.round(h));
});
ipcMain.handle("toggle-click-through", () => {
  // Click-through lets the table stay usable underneath; the window border
  // stays visible and Cmd/Ctrl+Shift+O always restores interaction.
  clickThrough = !clickThrough;
  win?.setIgnoreMouseEvents(clickThrough, { forward: true });
  return clickThrough;
});
ipcMain.handle("hide", () => win?.hide());
ipcMain.handle("quit", () => app.quit());

app.whenReady().then(async () => {
  try {
    engine = await startEngine(log);
  } catch (e: any) {
    log(`ENGINE UNAVAILABLE: ${e?.message ?? e}`);
    engine = null;
  }
  if (SMOKE) {
    const ok = engine !== null;
    console.log(ok ? "SMOKE OK" : "SMOKE FAIL: engine unavailable");
    if (engine) stopEngine(engine, log);
    app.exit(ok ? 0 : 1);
    return;
  }
  createWindow();
  // The restore chord always re-enables interaction, even in click-through.
  globalShortcut.register("CommandOrControl+Shift+O", () => {
    if (clickThrough) {
      clickThrough = false;
      win?.setIgnoreMouseEvents(false);
    } else {
      win?.isVisible() ? win.hide() : win?.show();
    }
  });
});

app.on("will-quit", () => {
  globalShortcut.unregisterAll();
  if (engine) stopEngine(engine, log);
});

app.on("window-all-closed", () => app.quit());
