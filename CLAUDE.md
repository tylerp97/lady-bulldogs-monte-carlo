# Lady Bulldogs Hoops — Coach Dashboard

Streamlit app for the Highland Bulldogs girls basketball coaching staff.
Pulls live data from stats.stlhighschoolsports.com.

## Running the app

```powershell
# First time only
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Every time
streamlit run app.py
```

## Running Python in this repo

Project dependencies live in `.venv`, **not** in the system interpreter. A bare
`python` will fail with `ModuleNotFoundError: No module named 'pandas'`.

```bash
.venv/Scripts/python.exe -c "import scraper; print(len(scraper.get_schedule()))"
```

The one exception is `.claude/hooks/check_python_syntax.py`, which is stdlib-only and
runs under whichever `python` is on PATH.

## Project constants

| Key | Value |
|-----|-------|
| Team ID | 111 |
| Season ID | 961 |
| Base URL | https://stats.stlhighschoolsports.com/sports/basketballgirls/stats |

Defined once in `scraper.py` (`TEAM_ID`, `SEASON_ID`, `BASE_URL`). Import them; never
re-hardcode these values in another module.

## Key pages

- Schedule: `teamschedule.php?s=961&t=111`
- Box score: `boxscore.php?s=961&e={event_id}`
- Player: `teamstatplayer.php?t=111&s=961&p={player_id}`

## Architecture

Four modules, strictly layered. Each layer may import from the ones above it, never below.

| Module | Responsibility | Must NOT |
|--------|----------------|----------|
| `scraper.py` | All HTTP + HTML parsing. Returns DataFrames. | Import streamlit |
| `simulation.py` | Monte Carlo matchup math + data-parity detection. | Import streamlit or scraper |
| `scouting.py` | Win/loss split analysis + scouting blurb generation. | Import streamlit or scraper |
| `app.py` | All Streamlit UI, caching, layout. | Contain parsing or math logic |

`simulation.py` and `scouting.py` take DataFrames as arguments — they never fetch.
That keeps them unit-testable without network access.

## Scraping contracts (hard-won — do not "simplify" these)

**Schedule page columns:** `Date | Time | Opponent | Result | Score(boxscore link) | Record`
The boxscore link is on the **score** cell, not the opponent cell. `_parse_schedule_html()`
locates the row by finding the cell containing a `boxscore.php` link and indexes
relative to it — do not switch to fixed column indices.

**Score format is always winner-loser.** `W 49-36` means Highland 49, opponent 36.
`L 49-36` means Highland 36. The `result` column decides the assignment.

**Box score table layout varies by home/away.** There is no fixed "4 tables" structure:

| Highland is | Table order |
|-------------|-------------|
| Home | `[0]` quarters, `[1]` season records, `[2]` Highland off, `[3]` Highland def |
| Away | `[0]` season records, `[1]` Highland off, `[2]` Highland def, `[3-4]` opponent |

`get_boxscore()` detects the format by checking whether `dfs[0]` has quarter columns
(`'1','2','3','4'`). For non-Highland teams, `get_boxscore_for_team()` uses **content-based**
detection instead (looks for `Pts` / `RBS` columns) because when only one team enters stats
their tables land at `[1-2]` regardless of home/away. Do not replace content detection with
index math.

**Home/away for opponents** is inferred from the opponent string: a leading `at ` means the
team was the visitor. `_fix_opponent()` repairs missing spaces (`atPana` -> `at Pana`) —
run it before any `startswith("at ")` check.

## Box score column reference

**Offensive:** #, Name, Pts, FG (X-Y), FG%, 2FG (X-Y), 2F%, 3FG (X-Y), 3F%, FT (X-Y), FT%
**Defensive:** #, Name, RBS, OF, DF, AST, STL, TN, BK, FLS

`_clean_boxscore()` splits every `X-Y` column into `{col}_made` / `{col}_att` (Int64) and
drops the original. Downstream code should reference `FG_made` / `FG_att`, not `FG`.

## Adding a new opponent

1. Find the team's ID from a `teamschedule.php?s=961&t={id}` URL on the source site.
2. Add an entry to `OPPONENT_METADATA` in `scraper.py`.
3. Set `has_turnovers` / `has_fg_pct` to `False` for any stat that team does not track.

These flags are **not cosmetic**. They drive the data-parity banner in `app.py`, the
simulation disclaimer in `simulation.py`, and the synthetic insight cards in `scouting.py`.
A wrong flag produces a scouting report that silently compares unlike quantities.

## Tuning the simulation

`simulation._SOS_PPG_K` (currently `10.0`) is the one intentional magic number: points of
scoring-mean shift per strength-of-schedule level away from average (SOS 3). It is
calibrated by hand against known matchups — the derivation is in the comment block above it.
Update that comment whenever you change the value.

`RNG_SEED = 42` is fixed so the dashboard does not flicker between reruns. Keep it fixed.

## Notes

- `get_all_boxscores()` issues one HTTP request per game in the schedule — it is the
  slowest call in the app and is cached for 1 hour in `app.py` via `@st.cache_data(ttl=3600)`.
- Be a good citizen: a 0.4s delay (`_REQUEST_DELAY`) is baked in between requests. Do not
  remove it or parallelize the fetch loop.
- Scrapers fail silently by design: `get_all_boxscores()` and `get_opponent_all_boxscores()`
  skip games that error rather than raising, so one bad page cannot break the dashboard.
  When debugging missing data, that `except Exception: continue` is the first place to look.

## UI conventions (2026 design)

The dashboard follows a modernist design handoff. The rules that keep it coherent:

- **Every radius is 0.** One global CSS rule in `app.py` does most of the visual work.
- **No emoji, no icon circles, no shadowed cards.** Structure comes from 2px dividers
  and alignment. `scouting.py` still emits an `icon` field on each insight; the UI
  deliberately ignores it.
- **Two colors carry meaning.** Highland is always ink (`#201e1d`), the opponent is
  always accent red (`#ec3013`). No green — that rules out `st.metric`, whose delta
  arrows force green/red, so records render as markdown instead.
- **No Plotly.** The win-probability split is a two-div flex bar. The dependency was
  removed from `requirements.txt`; do not reintroduce it for a chart.
- **Theme lives in `.streamlit/config.toml`** and is committed. `.gitignore` ignores
  everything else under `.streamlit/` so local `secrets.toml` never ships.
- **Interactive = native widget, visual = one HTML block.** Selectboxes, tabs and the
  roster table are native Streamlit; the scoreline, ledger and card grids are HTML.
  Reproducing the shared-rule grids via `st.container(border=True)` would depend on
  Streamlit's internal `data-testid` DOM, which changes between releases.
- **Copy generation belongs in `scouting.py`.** Imperative directives, role tags and
  the head-to-head read are content, not layout. `app.py` only arranges and escapes —
  every data-derived string goes through `_esc()` before entering an HTML block.

## agent_lab/ — exam practice, not part of the dashboard

`agent_lab/` holds Claude Certified Architect Foundations practice work. It is
deliberately isolated: it imports `scraper` only through explicit helpers
(`Grounder.from_schedule`), and nothing in `app.py`, `simulation.py`, `scouting.py`,
or `scraper.py` imports from it. Its dependencies live in `requirements-agent-lab.txt`
so the dashboard's runtime footprint does not grow.

Deleting the whole directory would leave the dashboard working. See `agent_lab/README.md`.
