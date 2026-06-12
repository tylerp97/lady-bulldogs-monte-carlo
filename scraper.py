import re
import time
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://stats.stlhighschoolsports.com/sports/basketballgirls/stats"
TEAM_ID = 111
SEASON_ID = 961
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; LadyBulldogsCoachApp/1.0)"}
_REQUEST_DELAY = 0.4  # seconds between requests — be a good citizen

KNOWN_OPPONENTS = {
    "Triad": 266,
    "Civic Memorial": 17,
    "Breese Central": 24,
    "Alton": 3,
}


def _get_html(path: str) -> str:
    resp = requests.get(f"{BASE_URL}/{path}", headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


def _fix_opponent(name: str) -> str:
    """Add a space after leading 'at' or 'vs' if one is missing (e.g. 'atPana' → 'at Pana')."""
    return re.sub(r'^(at|vs)(?=[^\s])', r'\1 ', name)


def _parse_schedule_html(html: str) -> pd.DataFrame:
    """
    Shared schedule-parsing logic for any team's schedule page.

    Page structure: Date | Time | Opponent | Result | Score(boxscore link) | Record
    Score format is always winner-loser, so W 49-36 means this team scored 49.
    """
    soup = BeautifulSoup(html, "lxml")
    rows = []

    for tr in soup.select("table tr"):
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue

        link_idx = next(
            (i for i, td in enumerate(tds)
             if td.find("a", href=lambda h: h and "boxscore.php" in h)),
            None,
        )
        if link_idx is None:
            continue

        link = tds[link_idx].find("a")
        event_id = int(link["href"].split("e=")[-1])
        score_text = tds[link_idx].get_text(strip=True)
        result = tds[link_idx - 1].get_text(strip=True)
        record = tds[link_idx + 1].get_text(strip=True) if link_idx + 1 < len(tds) else ""
        opponent = _fix_opponent(tds[2].get_text(separator=" ", strip=True))

        hld_score = opp_score = None
        if "-" in score_text and result in ("W", "L"):
            a, b = (int(x) for x in score_text.split("-", 1))
            hld_score, opp_score = (a, b) if result == "W" else (b, a)

        rows.append({
            "date": tds[0].get_text(strip=True),
            "opponent": opponent,
            "result": result,
            "hld_score": hld_score,
            "opp_score": opp_score,
            "margin": (hld_score - opp_score) if hld_score is not None else None,
            "record": record,
            "event_id": event_id,
        })

    return pd.DataFrame(rows)


def get_team_schedule(team_id: int, season_id: int = SEASON_ID) -> pd.DataFrame:
    """Fetch the season schedule for any team."""
    html = _get_html(f"teamschedule.php?s={season_id}&t={team_id}")
    return _parse_schedule_html(html)


def get_schedule() -> pd.DataFrame:
    """Fetch Highland's season schedule."""
    return get_team_schedule(TEAM_ID)


def _is_quarter_scores_table(df: pd.DataFrame) -> bool:
    """True if this table contains per-quarter scores (columns include '1','2','3','4')."""
    cols = {str(c) for c in df.columns}
    return {"1", "2", "3", "4"}.issubset(cols)


def _get_player_table_pairs(
    dfs: list[pd.DataFrame],
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Identify (offense, defense) pairs from box score tables by column content.
    Offense tables have a 'Pts' column; defense tables have 'RBS'.
    Returns pairs in page order — visitor first, home second when both are present.
    """
    pairs: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    pending_offense: pd.DataFrame | None = None

    for df in dfs:
        cols = {str(c) for c in df.columns}
        if "Pts" in cols and "Name" in cols:
            pending_offense = df
        elif "RBS" in cols and "Name" in cols and pending_offense is not None:
            pairs.append((pending_offense, df))
            pending_offense = None

    return pairs


def get_boxscore(event_id: int) -> dict[str, pd.DataFrame]:
    """
    Fetch Highland's box score for one game.

    Box score page structure varies by home/away:
      Home game  — [0] quarters, [1] season records, [2] Highland off, [3] Highland def
      Away game  — [0] season records, [1] Highland off, [2] Highland def, [3-4] opponent
    Detection: if dfs[0] has quarter columns ('1','2','3','4') → home format.
    """
    html = _get_html(f"boxscore.php?s={SEASON_ID}&e={event_id}")
    dfs = pd.read_html(StringIO(html))

    if _is_quarter_scores_table(dfs[0]):
        # Home format: Highland stats at [2] and [3]
        off_idx, def_idx = 2, 3
    else:
        # Away format: Highland (visitor) stats at [1] and [2]
        off_idx, def_idx = 1, 2

    offense = _clean_boxscore(dfs[off_idx]) if len(dfs) > off_idx else pd.DataFrame()
    defense = _clean_boxscore(dfs[def_idx]) if len(dfs) > def_idx else pd.DataFrame()
    return {"offense": offense, "defense": defense}


def get_boxscore_for_team(event_id: int, is_home: bool) -> dict[str, pd.DataFrame]:
    """
    Fetch box score stats for a non-Highland team from one of their game events.

    Uses content-based table detection (looks for 'Pts'/'RBS' columns) instead of
    fixed indices, because when only one team enters stats their tables are always
    at positions [1-2] regardless of home/away.

    When both teams entered stats: visitor pair is first, home pair is second.
    When only one team entered: that pair is returned regardless of is_home.
    """
    try:
        html = _get_html(f"boxscore.php?s={SEASON_ID}&e={event_id}")
        dfs = pd.read_html(StringIO(html))
    except Exception:
        return {"offense": pd.DataFrame(), "defense": pd.DataFrame()}

    pairs = _get_player_table_pairs(dfs)

    if not pairs:
        return {"offense": pd.DataFrame(), "defense": pd.DataFrame()}

    if len(pairs) == 1:
        off_raw, def_raw = pairs[0]
    else:
        # Visitor is first pair, home is second pair
        off_raw, def_raw = pairs[-1] if is_home else pairs[0]

    return {
        "offense": _clean_boxscore(off_raw),
        "defense": _clean_boxscore(def_raw),
    }


def _clean_boxscore(df: pd.DataFrame) -> pd.DataFrame:
    """Remove header/totals rows and split 'X-Y' shot columns into made/att pairs."""
    name_col = df.columns[1]
    df = df[~df[name_col].astype(str).isin(["—", "Name", "Totals"])].copy()
    df = df.reset_index(drop=True)

    shot_pattern = re.compile(r"^\d+-\d+$")
    for col in df.columns:
        sample = df[col].dropna().astype(str)
        if sample.empty:
            continue
        if sample.apply(lambda v: bool(shot_pattern.match(v))).mean() > 0.5:
            parts = df[col].astype(str).str.extract(r"^(\d+)-(\d+)$")
            df[f"{col}_made"] = pd.to_numeric(parts[0], errors="coerce").astype("Int64")
            df[f"{col}_att"] = pd.to_numeric(parts[1], errors="coerce").astype("Int64")
            df = df.drop(columns=[col])

    return df


def get_all_boxscores(schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch box scores for every game in Highland's schedule.

    Returns (offense_df, defense_df) with game-context columns added.
    Skips games that fail to load without raising.
    """
    off_frames: list[pd.DataFrame] = []
    def_frames: list[pd.DataFrame] = []

    for _, game in schedule.iterrows():
        try:
            bs = get_boxscore(int(game["event_id"]))
            for df, frames in [(bs["offense"], off_frames), (bs["defense"], def_frames)]:
                if df.empty:
                    continue
                enriched = df.copy()
                enriched["date"] = game["date"]
                enriched["opponent"] = game["opponent"]
                enriched["result"] = game["result"]
                frames.append(enriched)
            time.sleep(_REQUEST_DELAY)
        except Exception:
            continue

    off = pd.concat(off_frames, ignore_index=True) if off_frames else pd.DataFrame()
    def_ = pd.concat(def_frames, ignore_index=True) if def_frames else pd.DataFrame()
    return off, def_


def get_opponent_all_boxscores(schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch box scores for every game in an opponent's schedule.

    Uses get_boxscore_for_team() with home/away detection: if the opponent column
    starts with 'at ', the team was away (visitor tables [1-2]); otherwise home ([3-4]).
    Games where the team's stats aren't on the page are silently skipped.
    """
    off_frames: list[pd.DataFrame] = []
    def_frames: list[pd.DataFrame] = []

    for _, game in schedule.iterrows():
        try:
            is_home = not str(game["opponent"]).lower().startswith("at ")
            bs = get_boxscore_for_team(int(game["event_id"]), is_home)
            for df, frames in [(bs["offense"], off_frames), (bs["defense"], def_frames)]:
                if df.empty:
                    continue
                enriched = df.copy()
                enriched["date"] = game["date"]
                enriched["opponent"] = game["opponent"]
                enriched["result"] = game["result"]
                frames.append(enriched)
            time.sleep(_REQUEST_DELAY)
        except Exception:
            continue

    off = pd.concat(off_frames, ignore_index=True) if off_frames else pd.DataFrame()
    def_ = pd.concat(def_frames, ignore_index=True) if def_frames else pd.DataFrame()
    return off, def_


def get_opponent_season_data(team_id: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch the full season dataset for a known opponent:
      (schedule, offense_df, defense_df)

    offense_df / defense_df are the per-game individual box score rows with
    'date', 'opponent', 'result' context columns appended.
    """
    schedule = get_team_schedule(team_id)
    off_df, def_df = get_opponent_all_boxscores(schedule)
    return schedule, off_df, def_df


def aggregate_player_stats(off_df: pd.DataFrame, def_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine offensive and defensive box score data into per-player season totals + averages.

    The second column of each DataFrame is the player name.
    """
    if off_df.empty:
        return pd.DataFrame()

    def _sum_numeric(df: pd.DataFrame) -> pd.DataFrame:
        name_col = df.columns[1]
        numeric = df.select_dtypes(include="number").columns.tolist()
        gp = df.groupby(name_col).size().rename("GP")
        totals = df.groupby(name_col)[numeric].sum()
        return totals.join(gp)

    off_agg = _sum_numeric(off_df)
    def_agg = _sum_numeric(def_df) if not def_df.empty else pd.DataFrame()

    combined = off_agg.join(def_agg, how="left", rsuffix="_def")

    gp = combined["GP"]
    if "Pts" in combined.columns:
        combined["PPG"] = (combined["Pts"] / gp).round(1)
    for raw, avg_name in [("RBS", "RPG"), ("AST", "APG"), ("STL", "SPG")]:
        if raw in combined.columns:
            combined[avg_name] = (combined[raw] / gp).round(1)

    for base, pct_col in [("FG", "FG%"), ("3FG", "3FG%"), ("FT", "FT%")]:
        made_col, att_col = f"{base}_made", f"{base}_att"
        if made_col in combined.columns and att_col in combined.columns:
            combined[pct_col] = (
                combined[made_col] / combined[att_col].replace(0, pd.NA) * 100
            ).round(1)

    return combined.sort_values("Pts", ascending=False) if "Pts" in combined.columns else combined
