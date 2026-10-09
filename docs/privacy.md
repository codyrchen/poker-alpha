# Privacy and local data (Phase 72)

What PokerAlpha's code does with captured screen data. This describes the
code in this repository only; it is not a guarantee about your operating
system, browser, Python packages or anything else on your computer.

## Network

* No module in `poker_alpha/` or `experiments/` imports an HTTP / socket /
  mail client or calls any web API (checked by `tests/test_privacy.py`).
  No telemetry or analytics code exists in PokerAlpha.
* The observer, recorder, replay, annotation, export, debug report and
  decision engine read and write local files only.
* **Streamlit itself** (a third-party package) has two defaults that matter:
  1. it listens on **all network interfaces**, so other devices on your
     network could open the page — which shows captured frames;
  2. it sends **anonymous usage statistics** to Streamlit (page counts, not
     your frames).

  PokerAlpha ships `.streamlit/config.toml` (used when Streamlit is started
  from the repository root) with `server.address = "127.0.0.1"` and
  `browser.gatherUsageStats = false`; `./scripts/run_live_observer.sh` passes
  the same flags explicitly; and the app shows a **Privacy** warning at the
  top of the page if the running server is reachable from the network or has
  usage statistics on. Start it as documented and the page is reachable from
  this computer only.
* The Streamlit page sends the captured frames to your local browser to
  display them; with the settings above that connection stays on this
  computer.

## Local storage

| what | when | where | in Git? |
| --- | --- | --- | --- |
| single captured frame | only when you press **Save current frame** | `~/pokeralpha_captures/` (folder you choose) | no: the folder gets its own `*` `.gitignore`; `pokeralpha_captures/` is also in the repo `.gitignore` |
| observer test session | only while a test session records, and only kept frames (policy + limits: 4 h / 2,000 frames / 2 GB by default) | `~/pokeralpha_sessions/<id>/` | no: each session folder has a `*` `.gitignore`; `pokeralpha_sessions/` is in the repo `.gitignore` |
| debug report (JSON + HTML with region crops) | only when you press **Export current debug report** | `<save folder>/debug/` | no (`*` `.gitignore`) |
| fixture export | only when you run `experiments/export_observer_fixtures.py` | the `--output` folder you choose (refuses to overwrite) | only if you add it yourself |
| calibration JSON | when you press **Save calibration** | the path you give | only if you add it yourself |

Nothing is written for frames that are not kept, nothing is deleted or
overwritten automatically, and nothing is committed or uploaded by any
PokerAlpha command.

## Logs

PokerAlpha does not log frames or crops. Session `events.jsonl` and
`diagnostics/*.json` contain recognized values (cards, amounts, timings),
not images. Streamlit may print its own server log to the terminal.

## Before sharing data

Use `experiments/export_observer_fixtures.py`: it crops to the table and its
recognition regions (browser chrome and desktop removed), masks player-name
regions, and lists remaining text-like areas outside recognition regions for
you to check (`review_needed`) instead of masking them blindly. Open every
exported image before sending it anywhere.

## Repository

The only real screen data committed is
`tests/fixtures/pokernow/raw/hu_preflop_0001.png`, cropped to the table with
the chat preview and both player names painted over.

## Release / history audit (Phase 77)

`results/validation/release_audit.json`. All commits on the branch were
scanned for credential patterns, e-mail addresses, user-machine paths and
images: no secrets, no personal paths, e-mail only in commit metadata. The
only real screen data in history is `hu_preflop_0001.png` (one version).
**Open item for the owner:** that frame still shows the table's own
branding ("Squads with Quads $1,000 Freeroll" and the PokerNow logo). It is
not personal data but identifies the game; removing it from history would
need a force-push, which only the repository owner should decide. If this
frame came from a real-prize game, note that PokerAlpha's live mode is meant
only for games where outside assistance is allowed.
