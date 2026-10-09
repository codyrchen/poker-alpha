# Real PokerNow test checklist

Only in private / play-money / test games where outside assistance is
allowed. PokerAlpha only reads the screen; it never clicks or acts.

## Before the test (10 minutes)

1. `git fetch && git checkout claude/live-observer && git pull`
2. `python -m pip install -e ".[vision,ui]"` (once per environment)
3. `python -m poker_alpha.doctor` — every line OK (a WARN on screen capture
   on macOS = grant Screen Recording to your terminal app, then restart it)
4. `./scripts/run_live_observer.sh` -> browser opens `http://localhost:8501`
5. Input -> **Live screen** -> Mode **Observer Test Session (diagnostic)**
6. Monitor: the display with the PokerNow window. Capture rectangle: Cmd+Shift+4
   at the window's top-left and bottom-right corners (points), so the whole
   table **and** both player plates are inside with some margin. Press
   **Capture one frame**: the raw frame should show the table only.
7. Calibration: Layout **PokerNow Heads-Up** -> hero plate side (right / left)
   -> **Use PokerNow Heads-Up layout**. Bounds: **PokerNow felt (hue)**. The
   overlay's boxes should sit on cards, stacks, bets, the D button and the pot.
8. Blinds: enter the table's blinds. Browser zoom: leave at 100% for the whole
   session (write it down if not).
9. **Compute decisions: OFF** (default in test sessions).
10. Keep-frames policy: **hybrid (default)**; limits as default.
11. **Start session.**

## During the test (play 20-40 hands)

* Cover all streets (preflop, flop, turn, river) and showdowns.
* Both dealer positions (hero on the button and in the big blind).
* Card / suit diversity: hands with every suit, tens, aces, face cards.
* Different numbers: small and large bets, all-ins, a few stack sizes
  (decimals and integers).
* When something looks wrong on the page (wrong card, missing bet, phantom
  new hand): **Mark current frame** with a one-line note.
* Do not change browser zoom, window size or the capture rectangle mid-way
  (if you must: **Add note** saying so first).
* If PokerNow dims folded cards, fold a few hands on purpose (the observer
  cannot yet see the hero's own fold — this would help fix it).

## After the test

1. **Stop session** -> **Finish session** (the path is shown at the top).
2. Input -> **Annotate session**: annotate 20-50 representative frames
   (blank = unknown is fine). Set **Dataset role = held_out** for frames you
   did not tune anything on (that is all of them, for a first session).
3. Replay check: `python -m poker_alpha.observer.session_replay ~/pokeralpha_sessions/<id>`
   should print `reproduced: True`.
4. Export: `python experiments/export_observer_fixtures.py --session ~/pokeralpha_sessions/<id> --output ~/pokeralpha_fixture_export --preview`,
   look at `~/pokeralpha_fixture_export/preview/*.png`, then run it again
   without `--preview` (new output folder). Open every exported image.
5. Metrics: `python experiments/observer_validation.py --fixture-dir ~/pokeralpha_sessions/<id> --out ~/observer_validation.json`
6. Debug report of one bad frame: Live screen -> Region debugger -> **Export
   current debug report** (optional).

## What to send Claude

* `~/pokeralpha_sessions/<id>/manifest.json` and `session.json`
* the anonymized export folder (`raw/`, `annotations/`, `manifest.json`,
  `calibration.json`) — after you have looked at every image
* `~/observer_validation.json`
* a debug report (HTML + JSON) if you exported one
* full screenshots only if something cannot be understood otherwise
  (crop out tabs, chat and anything personal first)
