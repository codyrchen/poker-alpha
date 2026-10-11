# PyInstaller spec: the local PokerAlpha engine (API sidecar) with the
# release strategy + confidence artifacts and bundled demo data.
# Inference-time needs only — no training checkpoints, no screenshots.
from PyInstaller.utils.hooks import collect_submodules

a = Analysis(
    ["api_entry.py"],
    pathex=[".."],
    datas=[
        ("../results/strategy/holdem_v2_native_seed0_2m.npz", "results/strategy"),
        ("../results/strategy/holdem_v2_native_seed0_2m_confidence.npz", "results/strategy"),
        ("../poker_alpha/data", "poker_alpha/data"),
    ],
    hiddenimports=(collect_submodules("poker_alpha")
                   + ["uvicorn.logging", "uvicorn.loops.auto",
                      "uvicorn.protocols.http.auto",
                      "uvicorn.protocols.websockets.auto",
                      "uvicorn.lifespan.on"]),
    excludes=["matplotlib", "tkinter", "PyQt5", "IPython", "pytest",
              "streamlit", "PIL", "mss", "pytesseract"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, exclude_binaries=True, name="pokeralpha-engine",
          console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="pokeralpha-engine")
