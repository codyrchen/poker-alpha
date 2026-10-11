import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("pokeralpha", {
  engineBase: (): Promise<string | null> => ipcRenderer.invoke("engine-base"),
  engineLogs: (): Promise<string[]> => ipcRenderer.invoke("engine-logs"),
  setSize: (w: number, h: number): Promise<void> =>
    ipcRenderer.invoke("set-size", w, h),
  toggleClickThrough: (): Promise<boolean> =>
    ipcRenderer.invoke("toggle-click-through"),
  hide: (): Promise<void> => ipcRenderer.invoke("hide"),
  quit: (): Promise<void> => ipcRenderer.invoke("quit"),
});
