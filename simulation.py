import numpy as np
import pandas as pd

N_SIMS = 10_000
RNG_SEED = 42

# Highland always tracks the full set — used to detect opponent data gaps
_HIGHLAND_CATS = {"scoring", "shooting", "turnovers"}

_STAT_LABELS = {
    "shooting": "shooting percentages (FG%, 3FG%, FT%)",
    "turnovers": "individual turnovers (TN column)",
}


def detect_opponent_stat_categories(
    off_df: pd.DataFrame | None,
    def_df: pd.DataFrame | None,
) -> set[str]:
    """Return the set of stat categories available in an opponent's scraped data."""
    available = {"scoring"}

    if off_df is not None and not off_df.empty:
        cols = {str(c) for c in off_df.columns}
        if any("FG" in c and "made" in c for c in cols):
            available.add("shooting")

    if def_df is not None and not def_df.empty:
        cols = {str(c) for c in def_df.columns}
        if "TN" in cols:
            available.add("turnovers")

    return available


def build_disclaimer(opponent_name: str, opp_cats: set[str]) -> str | None:
    """
    Return a markdown disclaimer string if the opponent is missing stats Highland tracks.
    Returns None when both teams have full data parity.
    """
    missing = _HIGHLAND_CATS - opp_cats
    if not missing:
        return None

    lines = [f"- {_STAT_LABELS.get(cat, cat)}" for cat in sorted(missing)]
    return (
        f"**Data parity notice — {opponent_name}:** "
        f"The following stats are not tracked by {opponent_name} "
        f"and have been excluded from this simulation for a fair comparison:\n\n"
        + "\n".join(lines)
        + "\n\nThe simulation uses historical scoring distributions only."
    )


def run_monte_carlo(
    hld_scores: list[float],
    opp_scores: list[float],
    n_sims: int = N_SIMS,
) -> dict:
    """
    Simulate n_sims games by sampling from each team's historical scoring distribution.
    Both distributions are modelled as Normal(mean, std) and clipped to realistic range.
    """
    rng = np.random.default_rng(RNG_SEED)

    hld = np.array([s for s in hld_scores if s is not None and not np.isnan(float(s))], dtype=float)
    opp = np.array([s for s in opp_scores if s is not None and not np.isnan(float(s))], dtype=float)

    hld_mean = float(np.mean(hld))
    hld_std = float(max(np.std(hld), 2.0))
    opp_mean = float(np.mean(opp))
    opp_std = float(max(np.std(opp), 2.0))

    hld_sim = np.clip(rng.normal(hld_mean, hld_std, n_sims), 10, 110)
    opp_sim = np.clip(rng.normal(opp_mean, opp_std, n_sims), 10, 110)

    return {
        "win_prob": float(np.mean(hld_sim > opp_sim)),
        "loss_prob": float(np.mean(opp_sim > hld_sim)),
        "hld_projected": int(round(hld_mean)),
        "opp_projected": int(round(opp_mean)),
        "hld_mean": round(hld_mean, 1),
        "opp_mean": round(opp_mean, 1),
        "hld_std": round(hld_std, 1),
        "opp_std": round(opp_std, 1),
        "hld_n": len(hld),
        "opp_n": len(opp),
        "n_sims": n_sims,
    }


def simulate_matchup(
    hld_schedule: pd.DataFrame,
    opp_schedule: pd.DataFrame,
    opp_off_df: pd.DataFrame | None,
    opp_def_df: pd.DataFrame | None,
    opponent_name: str,
) -> dict:
    """
    Full simulation pipeline for a Highland vs. opponent matchup.

    Returns a dict with keys:
      simulation  — run_monte_carlo output
      disclaimer  — markdown string or None
      opp_stats   — sorted list of stat categories detected for the opponent
    """
    opp_cats = detect_opponent_stat_categories(opp_off_df, opp_def_df)
    disclaimer = build_disclaimer(opponent_name, opp_cats)

    hld_scores = hld_schedule["hld_score"].dropna().tolist()
    opp_scores = opp_schedule["hld_score"].dropna().tolist()

    sim = run_monte_carlo(hld_scores, opp_scores)

    return {
        "simulation": sim,
        "disclaimer": disclaimer,
        "opp_stats": sorted(opp_cats),
    }
