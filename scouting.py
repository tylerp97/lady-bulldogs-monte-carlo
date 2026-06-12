import pandas as pd
import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────────────

def _record(result_series: pd.Series) -> str:
    w = int((result_series == "W").sum())
    l = int((result_series == "L").sum())
    return f"{w}–{l}"


def _find_best_split(
    values: pd.Series,
    wins: pd.Series,
    min_each_side: int = 3,
) -> tuple[float, float, float, float]:
    """
    Scan 30th–70th percentile thresholds to find the one that maximises the
    win-rate differential between the two halves.

    Returns (threshold, wr_above, wr_below, diff).
    """
    best_diff = 0.0
    best_thresh = float(values.median())
    best_wr_above = best_wr_below = 0.5

    for pct in range(30, 71, 5):
        thresh = float(values.quantile(pct / 100))
        above = wins[values >= thresh]
        below = wins[values < thresh]
        if len(above) < min_each_side or len(below) < min_each_side:
            continue
        wa = float(above.mean())
        wb = float(below.mean())
        diff = abs(wa - wb)
        if diff > best_diff:
            best_diff, best_thresh, best_wr_above, best_wr_below = diff, thresh, wa, wb

    return best_thresh, best_wr_above, best_wr_below, best_diff


def _score_split_insight(
    schedule: pd.DataFrame,
    col: str,
    category: str,
    icon: str,
    min_diff: float = 0.25,
) -> dict | None:
    """Build a win/loss split insight for any numeric column in the schedule."""
    played = schedule[schedule["result"].isin(["W", "L"])].dropna(subset=[col]).copy()
    if len(played) < 6:
        return None

    wins = (played["result"] == "W")
    thresh, wr_above, wr_below, diff = _find_best_split(played[col], wins)
    if diff < min_diff:
        return None

    t = int(round(thresh))
    above = played[played[col] >= t]
    below = played[played[col] < t]

    if wr_above >= wr_below:
        headline = f"{t}+ pts → {wr_above:.0%} win rate"
        good_record, bad_record = _record(above["result"]), _record(below["result"])
        good_label, bad_label = f"≥ {t}", f"< {t}"
    else:
        headline = f"Under {t} pts → {wr_below:.0%} win rate"
        good_record, bad_record = _record(below["result"]), _record(above["result"])
        good_label, bad_label = f"< {t}", f"≥ {t}"

    return {
        "icon": icon,
        "category": category,
        "headline": headline,
        "threshold": t,
        "good_record": good_record,
        "bad_record": bad_record,
        "good_label": good_label,
        "bad_label": bad_label,
        "wr_above": wr_above,
        "wr_below": wr_below,
        "significance": diff,
    }


# ── Insight generators ─────────────────────────────────────────────────────────

def _turnover_insight(schedule: pd.DataFrame, def_df: pd.DataFrame) -> dict | None:
    """Win/loss split based on team turnovers per game (requires TN in def_df)."""
    if def_df is None or def_df.empty:
        return None
    if "TN" not in def_df.columns or "date" not in def_df.columns:
        return None

    tn_pg = def_df.groupby("date")["TN"].sum().reset_index()
    tn_pg.columns = ["date", "team_tn"]

    played = (
        schedule[schedule["result"].isin(["W", "L"])]
        .merge(tn_pg, on="date", how="inner")
    )
    if len(played) < 6:
        return None

    wins = (played["result"] == "W")
    thresh, wr_above, wr_below, diff = _find_best_split(played["team_tn"], wins, min_each_side=3)
    if diff < 0.20:
        return None

    t = int(round(thresh))
    above = played[played["team_tn"] >= t]
    below = played[played["team_tn"] < t]

    if wr_below > wr_above:
        headline = f"Under {t} TOs → {wr_below:.0%} win rate"
        good_record, bad_record = _record(below["result"]), _record(above["result"])
        good_label, bad_label = f"< {t} TOs", f"≥ {t} TOs"
    else:
        headline = f"{t}+ TOs → {wr_above:.0%} win rate"
        good_record, bad_record = _record(above["result"]), _record(below["result"])
        good_label, bad_label = f"≥ {t} TOs", f"< {t} TOs"

    return {
        "icon": "🔄",
        "category": "Ball Security",
        "headline": headline,
        "threshold": t,
        "good_record": good_record,
        "bad_record": bad_record,
        "good_label": good_label,
        "bad_label": bad_label,
        "wr_above": wr_above,
        "wr_below": wr_below,
        "significance": diff,
    }


def _player_impact_for(
    schedule: pd.DataFrame,
    off_df: pd.DataFrame,
    player_name: str,
    name_col: str,
) -> dict | None:
    """Win/loss split for one specific player's scoring."""
    player_games = (
        off_df[off_df[name_col] == player_name][["date", "Pts"]]
        .rename(columns={"Pts": "player_pts"})
    )
    played = (
        schedule[schedule["result"].isin(["W", "L"])]
        .merge(player_games, on="date", how="inner")
    )
    if len(played) < 5:
        return None

    wins = (played["result"] == "W")
    thresh, wr_above, wr_below, diff = _find_best_split(played["player_pts"], wins)
    if diff < 0.22:
        return None

    t = int(round(thresh))
    above = played[played["player_pts"] >= t]
    below = played[played["player_pts"] < t]

    return {
        "name": player_name,
        "season_avg": round(float(played["player_pts"].mean()), 1),
        "threshold": t,
        "record_above": _record(above["result"]),
        "record_below": _record(below["result"]),
        "wr_above": wr_above,
        "wr_below": wr_below,
        "significance": diff,
    }


def get_player_impacts(
    schedule: pd.DataFrame,
    off_df: pd.DataFrame,
    n: int = 3,
) -> list[dict]:
    """Return win/loss insights for the top N scorers by season total points."""
    if off_df is None or off_df.empty or "Pts" not in off_df.columns:
        return []
    if "date" not in off_df.columns:
        return []

    name_col = off_df.columns[1]
    top_players = off_df.groupby(name_col)["Pts"].sum().nlargest(n).index

    impacts = []
    for player in top_players:
        impact = _player_impact_for(schedule, off_df, player, name_col)
        if impact:
            impacts.append(impact)

    return impacts


def _shooting_summary(off_df: pd.DataFrame) -> dict | None:
    """Season 3-point shooting summary, if data is available."""
    if off_df is None or off_df.empty or "date" not in off_df.columns:
        return None

    made_col = next(
        (c for c in off_df.columns if "3FG" in str(c) and "made" in str(c).lower()), None
    )
    att_col = next(
        (c for c in off_df.columns if "3FG" in str(c) and "att" in str(c).lower()), None
    )
    if not made_col or not att_col:
        return None

    total_made = int(off_df[made_col].sum())
    total_att = int(off_df[att_col].sum())
    n_games = off_df["date"].nunique()

    if total_att == 0 or n_games == 0:
        return None

    return {
        "pct": round(total_made / total_att * 100, 1),
        "made": total_made,
        "att": total_att,
        "att_per_game": round(total_att / n_games, 1),
        "made_per_game": round(total_made / n_games, 1),
    }


def _trend_summary(schedule: pd.DataFrame) -> dict:
    """Compare first-half vs second-half of season; surface last-5 form."""
    played = (
        schedule[schedule["result"].isin(["W", "L"])]
        .dropna(subset=["hld_score"])
        .reset_index(drop=True)
    )
    n = len(played)

    if n < 6:
        return {"n_games": n}

    mid = n // 2
    early = played.iloc[:mid]
    recent = played.iloc[mid:]
    last5 = played.iloc[-5:]

    early_wr = float((early["result"] == "W").mean())
    recent_wr = float((recent["result"] == "W").mean())

    return {
        "n_games": n,
        "season_record": _record(played["result"]),
        "season_ppg": round(float(played["hld_score"].mean()), 1),
        "season_opp_ppg": round(float(played["opp_score"].mean()), 1),
        "early_win_rate": round(early_wr, 3),
        "recent_win_rate": round(recent_wr, 3),
        "early_ppg": round(float(early["hld_score"].mean()), 1),
        "recent_ppg": round(float(recent["hld_score"].mean()), 1),
        "last5_record": _record(last5["result"]),
        "last5_ppg": round(float(last5["hld_score"].mean()), 1),
        "trending_up": recent_wr > early_wr,
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_full_scouting_report(
    schedule: pd.DataFrame,
    off_df: pd.DataFrame | None,
    def_df: pd.DataFrame | None,
    opponent_name: str,
) -> dict:
    """
    Assemble the complete scouting report for an opponent.

    Returns a dict with keys: opponent_name, trend, insights, player_impacts, shooting.
    """
    insights: list[dict] = []

    s = _score_split_insight(schedule, "hld_score", "Points Scored", "🎯")
    if s:
        insights.append(s)

    s = _score_split_insight(schedule, "opp_score", "Points Allowed", "🛡️")
    if s:
        insights.append(s)

    s = _turnover_insight(schedule, def_df)
    if s:
        insights.append(s)

    insights.sort(key=lambda x: x["significance"], reverse=True)

    return {
        "opponent_name": opponent_name,
        "trend": _trend_summary(schedule),
        "insights": insights,
        "player_impacts": get_player_impacts(schedule, off_df, n=3),
        "shooting": _shooting_summary(off_df),
    }
