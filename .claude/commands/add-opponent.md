---
description: Add a new opponent to OPPONENT_METADATA with verified data-parity flags
argument-hint: <team name> [team id]
allowed-tools: Read, Edit, Grep, Bash
---

Add **$1** to the dashboard's opponent list.

Follow this exact order — the parity flags must be *observed*, never assumed:

1. Determine the team id. If `$2` was given, use it. Otherwise ask; do not guess an id.
2. Fetch the team's real data before editing anything:
   `.venv/Scripts/python.exe -c "import scraper; s,o,d = scraper.get_opponent_season_data($2); print(len(s)); print(sorted(o.columns)); print(sorted(d.columns))"`
3. Derive the flags from what actually came back:
   - `has_fg_pct = True` iff the offense frame has `FG_made` and `FG_att`
   - `has_turnovers = True` iff the defense frame has `TN`
4. Add the entry to `OPPONENT_METADATA` in `scraper.py`, keeping the existing comment
   style that explains *why* a flag is False.
5. Confirm `KNOWN_OPPONENTS` picks it up (it is derived, so it should need no edit) and
   that the name renders in the `app.py` opponent selectbox.

If the fetch returns an empty schedule or zero box scores, stop and report that — an
opponent with no parseable data should not be added, because the dashboard will show a
scouting report built on nothing.
