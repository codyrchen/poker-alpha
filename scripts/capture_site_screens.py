"""Capture the screenshots shown on the public site (apps/web/public/screens).

Each image is a real capture of the current static build, taken in a
headless browser at a 1280 x 800 viewport (light theme, 2x, downscaled). Interactive pages
are driven the way a visitor would: /learn runs 10,000 CFR iterations in
the page, /arena starts a match.

    cd apps/web && npm run build && cd ../..
    pip install playwright pillow && python -m playwright install chromium
    python scripts/capture_site_screens.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "apps" / "web" / "out"
DEST = ROOT / "apps" / "web" / "public" / "screens"
PORT = 3131
VIEWPORT = {"width": 1280, "height": 800}


def drive_learn(page) -> None:
    page.get_by_role("button", name="Run 10,000").click()
    page.wait_for_function(
        "() => Number((document.querySelector('.stat .v')?.textContent || '0').replace(/,/g, '')) >= 10000",
        timeout=60_000,
    )
    time.sleep(0.5)
    # Frame the exploitability readout and the curve, not the page header.
    page.evaluate("() => document.querySelector('h1').scrollIntoView()")


def drive_arena(page) -> None:
    page.get_by_role("button", name="Play 25 hands").click()
    time.sleep(0.5)


# (name, route, driver, card anchor). The full viewport goes to <name>.webp for
# the gallery; when an anchor is given, a 16:10 crop starting at that element
# goes to <name>-card.webp for the demo cards.
SHOTS = [
    ("home", "/", None, None),
    ("learn", "/learn", drive_learn, "() => document.querySelector('.status')"),
    ("arena", "/arena", drive_arena, "() => document.querySelector('h1')"),
    ("phase2", "/lab/phase2", None, "() => document.querySelectorAll('h2')[0]"),
    ("engineering", "/engineering", None, None),
]
CARD_X, CARD_W, CARD_H = 128, 760, 475


def main() -> int:
    if not (OUT_DIR / "index.html").exists():
        print("build the site first: cd apps/web && npm run build", file=sys.stderr)
        return 1
    DEST.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(PORT), "-d", str(OUT_DIR)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=2, color_scheme="light",
                                      reduced_motion="reduce")
            page = ctx.new_page()
            for name, route, drive, anchor in SHOTS:
                url = f"http://localhost:{PORT}{route if route == '/' else route + '.html'}"
                page.goto(url, wait_until="networkidle")
                if drive:
                    drive(page)
                png = page.screenshot(full_page=False)
                Image.open(BytesIO(png)).convert("RGB").resize((VIEWPORT["width"], VIEWPORT["height"]), Image.LANCZOS).save(DEST / f"{name}.webp", quality=82, method=6)
                print(f"{name}.webp", (DEST / f"{name}.webp").stat().st_size // 1024, "KB")
                if anchor:
                    top = page.evaluate(f"(() => {{ const el = ({anchor})(); return el.getBoundingClientRect().top + window.scrollY; }})()")
                    clip = {"x": CARD_X, "y": max(0, top - 20), "width": CARD_W, "height": CARD_H}
                    png = page.screenshot(full_page=True, clip=clip)
                    card = Image.open(BytesIO(png)).convert("RGB").resize((1000, 625), Image.LANCZOS)
                    card.save(DEST / f"{name}-card.webp", quality=85, method=6)
            browser.close()
    finally:
        server.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
