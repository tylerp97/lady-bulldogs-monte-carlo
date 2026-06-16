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


# ── Scouting blurb generators ──────────────────────────────────────────────────

def _insight_blurb(category: str, threshold: int, wr_above: float, wr_below: float) -> str:
    """Generate a basketball-expert scouting blurb for a statistical insight."""
    wr_high = max(wr_above, wr_below)
    decisive = wr_high > 0.75

    if category == "Points Scored":
        if threshold >= 50:
            return (
                "Explosive offense that thrives on open-court opportunities and scoring runs. "
                "Apply ball pressure early, minimize live-ball turnovers that fuel their transition game, "
                "and force deliberate half-court sets where your defense can dictate."
            )
        elif threshold >= 42:
            return (
                "Consistent offensive output anchors their wins. "
                "Disrupting their shooting rhythm — particularly on early-clock possessions — "
                "takes them out of their comfort zone and invites impatience."
            )
        else:
            return (
                "They build wins through efficiency and discipline, not volume. "
                "Match their pace, contest every touch in the paint, "
                "and limit second-chance opportunities to keep them uncomfortable."
            )

    elif category == "Points Allowed":
        if wr_below > wr_above:
            if decisive:
                return (
                    "Elite defensive unit that locks down opponents and makes every bucket feel earned. "
                    "Patient offense, ball movement through weak-side rotations, "
                    "and attacking in transition after live-ball turnovers are your best openings."
                )
            else:
                return (
                    "Above-average defensive discipline when locked in. "
                    "Ball movement, skip passes, and attacking late in the shot clock "
                    "can crack their coverage and force defensive breakdowns."
                )
        else:
            return (
                "Vulnerable in open-ended, high-scoring games. "
                "Attack early, build a lead, and maintain offensive aggression "
                "to prevent them from settling into any kind of defensive rhythm."
            )

    elif category == "Ball Security":
        if wr_below > wr_above:
            return (
                "Ball security is the heartbeat of their offense. "
                "Force weak-hand dribbles, trap ball screens, and apply full-court pressure "
                "to manufacture the chaos that completely derails their game plan."
            )
        else:
            return (
                "Surprisingly resilient despite sloppy possessions — "
                "their scoring volume absorbs turnover damage. "
                "Prioritize your own ball security and exploit the live-ball transition "
                "opportunities they consistently create."
            )

    return ""


def _player_blurb(
    season_avg: float,
    wr_above: float,
    wr_below: float,
    fg_pct: float | None = None,
) -> str:
    """Generate a basketball-expert scouting blurb for a player impact insight."""
    sig = abs(wr_above - wr_below)

    if season_avg >= 20:
        base = (
            "Primary offensive engine — the entire game plan runs through her. "
            "She demands a disciplined shadow defender, early ball denial on every entry, "
            "and constant help-side communication on all penetration."
        )
    elif season_avg >= 15:
        base = (
            "High-volume second option who can take over when the primary scorer is contained. "
            "Limit catch opportunities on the wing, deny easy post entries, "
            "and hedge conservatively on ball screens."
        )
    elif season_avg >= 10:
        base = (
            "Efficient role scorer who elevates in transition and off back-screens. "
            "Maintain disciplined off-ball awareness — she converts quiet opportunities "
            "into momentum-shifting buckets."
        )
    else:
        base = (
            "Spot-up threat whose catch-and-shoot game can tilt a quarter. "
            "Stay disciplined and don't overplay her — it opens driving lanes for the primary options."
        )

    if sig > 0.40:
        impact = (
            " Her scoring is the single strongest predictor of this team's outcomes — "
            "the win probability swings dramatically based on her production. Shutting her down is the #1 gameplan priority."
        )
    elif sig > 0.25:
        impact = (
            " Her production has a clear ripple effect on team success — "
            "early foul trouble or a cold start changes the entire complexion of the game."
        )
    else:
        impact = (
            " Consistent contributor, but the team has enough depth to compensate when she's quiet."
        )

    if fg_pct is not None:
        if fg_pct >= 50:
            eff = " High-efficiency finisher — physical defense inside and forcing baseline are critical."
        elif fg_pct >= 40:
            eff = " Solid efficiency across the floor — make her work hard outside her comfort zone."
        else:
            eff = " Volume scorer with inconsistent efficiency — force contested looks and she'll struggle."
        return base + impact + eff

    return base + impact


def _shooting_blurb(pct: float, att_per_game: float) -> str:
    """Generate a scouting blurb for 3-point shooting tendency."""
    if pct >= 40:
        return (
            f"Elite perimeter attack at {pct}% — they will beat you from three if you give any daylight. "
            "Stay attached on every catch, communicate through all ball screens, "
            "and prioritize a hand in every shooter's face regardless of position."
        )
    elif pct >= 33:
        return (
            f"Legitimate 3-point threat at {pct}%. Play one step off to contest "
            "while maintaining lane presence. "
            "Off-ball cutters become dangerous when defenders shade too far to the arc."
        )
    elif pct >= 25:
        return (
            f"Below-average efficiency from three at {pct}%. Play them honest but shade help-side — "
            "prioritize lane clogging and rebound positioning over tight perimeter coverage."
        )
    else:
        return (
            f"Weak from the perimeter at {pct}% — sag off and load the paint. "
            f"Force them to beat your defense off the bounce and challenge every interior attempt. "
            f"At {att_per_game} attempts per game they're still looking for it, but it's not a threat."
        )


# ── Statistical insight generators ────────────────────────────────────────────

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
        "blurb": _insight_blurb(category, t, wr_above, wr_below),
        "threshold": t,
        "good_record": good_record,
        "bad_record": bad_record,
        "good_label": good_label,
        "bad_label": bad_label,
        "wr_above": wr_above,
        "wr_below": wr_below,
        "significance": diff,
        "is_synthetic": False,
    }


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
        "blurb": _insight_blurb("Ball Security", t, wr_above, wr_below),
        "threshold": t,
        "good_record": good_record,
        "bad_record": bad_record,
        "good_label": good_label,
        "bad_label": bad_label,
        "wr_above": wr_above,
        "wr_below": wr_below,
        "significance": diff,
        "is_synthetic": False,
    }


def _player_impact_for(
    schedule: pd.DataFrame,
    off_df: pd.DataFrame,
    player_name: str,
    name_col: str,
) -> dict | None:
    """Win/loss split for one specific player's scoring, with scouting blurb."""
    player_rows = off_df[off_df[name_col] == player_name]
    player_games = (
        player_rows[["date", "Pts"]]
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
    season_avg = round(float(played["player_pts"].mean()), 1)

    # Pull FG% if available
    fg_pct = None
    if "FG_made" in off_df.columns and "FG_att" in off_df.columns:
        total_made = float(player_rows["FG_made"].sum())
        total_att = float(player_rows["FG_att"].sum())
        if total_att > 0:
            fg_pct = round(total_made / total_att * 100, 1)

    return {
        "name": player_name,
        "season_avg": season_avg,
        "threshold": t,
        "record_above": _record(above["result"]),
        "record_below": _record(below["result"]),
        "wr_above": wr_above,
        "wr_below": wr_below,
        "significance": diff,
        "fg_pct": fg_pct,
        "blurb": _player_blurb(season_avg, wr_above, wr_below, fg_pct),
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


# ── Season-level analysis ──────────────────────────────────────────────────────

def _shooting_summary(off_df: pd.DataFrame) -> dict | None:
    """Season 3-point shooting summary with scouting blurb, if data is available."""
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

    pct = round(total_made / total_att * 100, 1)
    att_per_game = round(total_att / n_games, 1)

    return {
        "pct": pct,
        "made": total_made,
        "att": total_att,
        "att_per_game": att_per_game,
        "made_per_game": round(total_made / n_games, 1),
        "blurb": _shooting_blurb(pct, att_per_game),
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


# ── Synthetic insights for missing data ───────────────────────────────────────

def generate_missing_data_insights(
    opponent_name: str,
    has_turnovers: bool,
    has_fg_pct: bool,
) -> list[dict]:
    """
    Generate expert-commentary insight cards for stats that Highland tracks
    but the opponent does not. These turn data gaps into actionable coaching notes.
    """
    insights = []

    if not has_turnovers:
        insights.append({
            "icon": "👻",
            "category": "Invisible Possession Battle",
            "headline": f"{opponent_name} doesn't track individual turnovers",
            "blurb": (
                "The possession battle — a key predictor of outcome in girls basketball — "
                "is an unquantified variable here. The simulation runs on scoring distributions only. "
                "Regardless of what the data shows, disciplined ball security and "
                "preventing live-ball turnovers will be critical to controlling pace "
                "and limiting their second-chance opportunities."
            ),
            "good_record": None,
            "bad_record": None,
            "good_label": None,
            "bad_label": None,
            "wr_above": None,
            "wr_below": None,
            "significance": 0,
            "is_synthetic": True,
        })

    if not has_fg_pct:
        insights.append({
            "icon": "🎯",
            "category": "Shooting Profile Unknown",
            "headline": f"FG%, 3FG%, and FT% not tracked for {opponent_name}",
            "blurb": (
                "Shooting efficiency is a blind spot in this scouting report — "
                "the simulation relies on raw scoring totals only. "
                "Contest all shots, prioritize defensive rebounding position, "
                "and don't let their scoring totals mislead you: "
                "some of it may be high-volume, low-efficiency production "
                "that won't hold up under tight pressure defense."
            ),
            "good_record": None,
            "bad_record": None,
            "good_label": None,
            "bad_label": None,
            "wr_above": None,
            "wr_below": None,
            "significance": 0,
            "is_synthetic": True,
        })

    return insights


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_full_scouting_report(
    schedule: pd.DataFrame,
    off_df: pd.DataFrame | None,
    def_df: pd.DataFrame | None,
    opponent_name: str,
    has_turnovers: bool = True,
    has_fg_pct: bool = True,
) -> dict:
    """
    Assemble the complete scouting report for an opponent.

    Returns a dict with keys: opponent_name, trend, insights, player_impacts, shooting.
    Each insight and player_impact includes a 'blurb' field with expert commentary.
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

    # Sort data-driven insights by significance, then append synthetic commentary
    insights.sort(key=lambda x: x["significance"], reverse=True)
    insights.extend(generate_missing_data_insights(opponent_name, has_turnovers, has_fg_pct))

    return {
        "opponent_name": opponent_name,
        "trend": _trend_summary(schedule),
        "insights": insights,
        "player_impacts": get_player_impacts(schedule, off_df, n=3),
        "shooting": _shooting_summary(off_df),
    }
