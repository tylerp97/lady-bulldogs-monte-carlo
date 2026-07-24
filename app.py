import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scraper import (
    KNOWN_OPPONENTS,
    OPPONENT_METADATA,
    aggregate_player_stats,
    get_opponent_season_data,
    get_schedule,
)
from simulation import simulate_matchup
from scouting import generate_full_scouting_report

st.set_page_config(
    page_title="Highland Lady Bulldogs",
    page_icon="🏀",
    layout="wide",
)

st.title("🏀 Highland Lady Bulldogs — 2025-26")

_SOS_LABELS = {
    1: "1 — Easiest",
    2: "2 — Easy",
    3: "3 — Average",
    4: "4 — Hard",
    5: "5 — Hardest",
}

# ── Team + SOS selectors ───────────────────────────────────────────────────────

col_my, col_opp = st.columns(2)
with col_my:
    st.selectbox("My Team", ["Highland"], key="my_team")
    hld_sos = st.selectbox(
        "Strength of Schedule",
        options=[1, 2, 3, 4, 5],
        index=2,
        format_func=lambda x: _SOS_LABELS[x],
        key="hld_sos",
        help="Rate the overall difficulty of Highland's schedule. Higher = tougher opponents faced.",
    )
with col_opp:
    opponent_name = st.selectbox("Opposing Team", list(KNOWN_OPPONENTS.keys()))
    opp_sos = st.selectbox(
        "Strength of Schedule",
        options=[1, 2, 3, 4, 5],
        index=2,
        format_func=lambda x: _SOS_LABELS[x],
        key="opp_sos",
        help=f"Rate the overall difficulty of this team's schedule.",
    )

meta = OPPONENT_METADATA[opponent_name]


# ── Cached data loaders ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_schedule() -> pd.DataFrame:
    return get_schedule()


@st.cache_data(ttl=3600, show_spinner=False)
def load_opponent_data(team_id: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return get_opponent_season_data(team_id)


# ── Load data ──────────────────────────────────────────────────────────────────

with st.spinner("Loading Highland schedule..."):
    hld_schedule = load_schedule()

hld_played = hld_schedule[hld_schedule["result"].isin(["W", "L"])]

with st.spinner(
    f"Loading {opponent_name} season data — first load ~15 s, then cached 1 hour..."
):
    opp_schedule, opp_off_df, opp_def_df = load_opponent_data(meta["team_id"])

opp_played = opp_schedule[opp_schedule["result"].isin(["W", "L"])]


# ── Dynamic data completeness banner ──────────────────────────────────────────

missing_items = []
if not meta["has_turnovers"]:
    missing_items.append("individual turnovers")
if not meta["has_fg_pct"]:
    missing_items.append("field goal percentages")

if missing_items:
    missing_str = " and ".join(missing_items)
    st.warning(
        f"**{opponent_name} is missing {missing_str}.** "
        f"These stats are excluded from simulations for both teams to keep comparisons fair."
    )
else:
    st.info(
        f"📊 **{opponent_name} has a complete dataset.** "
        f"Both teams track the same stats — scoring, shooting percentages, and turnovers."
    )


# ── Monte Carlo simulation ─────────────────────────────────────────────────────

matchup = simulate_matchup(
    hld_schedule=hld_played,
    opp_schedule=opp_played,
    opp_off_df=opp_off_df,
    opp_def_df=opp_def_df,
    opponent_name=opponent_name,
    hld_sos=hld_sos,
    opp_sos=opp_sos,
)
sim = matchup["simulation"]


# ── Simulation results ─────────────────────────────────────────────────────────

st.subheader(f"Highland vs. {opponent_name} — Win Probability")

col_donut, col_stats = st.columns([1, 1])

with col_donut:
    win_pct = sim["win_prob"]
    loss_pct = sim["loss_prob"]
    raw_pct = sim["raw_win_prob"]

    fig = go.Figure(
        go.Pie(
            values=[win_pct * 100, loss_pct * 100],
            labels=["Highland", opponent_name],
            hole=0.65,
            marker=dict(colors=["#f1c40f", "#9b59b6"]),
            textinfo="none",
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        )
    )
    fig.add_annotation(
        text=f"<b>{win_pct:.0%}</b>",
        x=0.5, y=0.58,
        font=dict(size=38, color="#f1c40f" if win_pct >= 0.5 else "#9b59b6"),
        showarrow=False,
    )
    fig.add_annotation(
        text="Highland win probability",
        x=0.5, y=0.42,
        font=dict(size=13, color="#888888"),
        showarrow=False,
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", y=-0.08),
        height=320,
        margin=dict(t=10, b=20, l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    if hld_sos != opp_sos:
        raw_hld = sim["raw_hld_projected"]
        raw_opp = sim["raw_opp_projected"]
        st.caption(
            f"Raw (no SOS): **{raw_pct:.0%}** — {raw_hld}–{raw_opp} · "
            f"SOS-adjusted: **{win_pct:.0%}** "
            f"(Highland SOS {hld_sos} vs {opponent_name} SOS {opp_sos})"
        )
    else:
        st.caption(
            f"Based on {sim['n_sims']:,} simulations · "
            f"Highland: {sim['hld_n']} games · {opponent_name}: {sim['opp_n']} games"
        )

with col_stats:
    st.markdown("#### Projected Score")
    c1, c2 = st.columns(2)
    c1.metric("Highland", sim["hld_projected"],
              delta=f"avg {sim['hld_mean']} ± {sim['hld_std']} pts")
    c2.metric(opponent_name, sim["opp_projected"],
              delta=f"avg {sim['opp_mean']} ± {sim['opp_std']} pts",
              delta_color="off")

    st.markdown("---")

    st.markdown("#### Head-to-Head This Season")
    h2h = hld_schedule[
        hld_schedule["opponent"].str.contains(opponent_name, case=False, na=False)
    ]
    if not h2h.empty:
        for _, g in h2h.iterrows():
            if pd.notna(g.get("hld_score")):
                if g["result"] == "W":
                    badge = f"✅ W  {int(g['hld_score'])}–{int(g['opp_score'])}"
                elif g["result"] == "L":
                    badge = f"❌ L  {int(g['hld_score'])}–{int(g['opp_score'])}"
                else:
                    badge = "—"
                st.markdown(f"**{g['date']}** — {badge}")
    else:
        st.caption("No completed head-to-head game found.")

    st.markdown("---")
    st.caption(
        f"Based on {sim['n_sims']:,} simulations · "
        f"Highland {sim['hld_n']} games · {opponent_name} {sim['opp_n']} games"
    )

st.divider()


# ── Scouting report ────────────────────────────────────────────────────────────

st.subheader(f"📋 Scouting Report — {opponent_name}")

report = generate_full_scouting_report(
    opp_schedule, opp_off_df, opp_def_df, opponent_name,
    has_turnovers=meta["has_turnovers"],
    has_fg_pct=meta["has_fg_pct"],
)
trend = report["trend"]

# Season overview strip
if trend.get("n_games", 0) >= 4:
    with st.container(border=True):
        st.caption(f"{opponent_name} — Season Overview ({trend['n_games']} games played)")
        oc1, oc2, oc3, oc4, oc5 = st.columns(5)
        oc1.metric("Record", trend.get("season_record", "—"))
        oc2.metric("PPG", trend.get("season_ppg", "—"))
        oc3.metric("Opp PPG", trend.get("season_opp_ppg", "—"))
        oc4.metric("Last 5", trend.get("last5_record", "—"))
        arrow = "↑" if trend.get("trending_up") else "↓"
        early_wr = trend.get("early_win_rate", 0)
        recent_wr = trend.get("recent_win_rate", 0)
        delta_pct = f"{arrow} {abs(recent_wr - early_wr):.0%} vs first half"
        oc5.metric("Form", f"{recent_wr:.0%} WR (2nd half)", delta=delta_pct)
else:
    st.info(f"Not enough completed games to generate a season overview for {opponent_name}.")

st.markdown("&nbsp;", unsafe_allow_html=True)


# ── Key Performance Patterns ───────────────────────────────────────────────────

insights = report["insights"]
if insights:
    st.markdown("##### Key Performance Patterns")
    for row_start in range(0, len(insights), 2):
        row = insights[row_start: row_start + 2]
        cols = st.columns(len(row))
        for col, ins in zip(cols, row):
            with col:
                with st.container(border=True):
                    st.markdown(f"## {ins['icon']}")
                    st.markdown(f"**{ins['category']}**")
                    st.markdown(f"*{ins['headline']}*")
                    if ins.get("blurb"):
                        st.markdown(ins["blurb"])
                    st.markdown("&nbsp;", unsafe_allow_html=True)
                    if not ins.get("is_synthetic"):
                        ic1, ic2 = st.columns(2)
                        ic1.metric(
                            f"✅ {ins['good_label']}",
                            ins["good_record"],
                            delta=f"{max(ins['wr_above'], ins['wr_below']):.0%} WR",
                        )
                        ic2.metric(
                            f"❌ {ins['bad_label']}",
                            ins["bad_record"],
                            delta=f"−{abs(ins['wr_above'] - ins['wr_below']):.0%}",
                            delta_color="inverse",
                        )
    st.markdown("&nbsp;", unsafe_allow_html=True)
else:
    st.info("Not enough game data to surface meaningful win/loss patterns.")


# ── Roster Breakdown ───────────────────────────────────────────────────────────

player_stats = aggregate_player_stats(opp_off_df, opp_def_df)
if not player_stats.empty:
    st.markdown("##### Roster Breakdown")
    priority_cols = ["GP", "PPG", "Pts", "FG%", "3FG%", "FT%", "RPG", "RBS", "APG", "AST", "STL", "SPG"]
    show_cols = [c for c in priority_cols if c in player_stats.columns]
    display_df = player_stats[show_cols].head(10).reset_index()
    display_df = display_df.rename(columns={display_df.columns[0]: "Player"})
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)


# ── Player Impact Analysis ─────────────────────────────────────────────────────

player_impacts = report["player_impacts"]
if player_impacts:
    st.markdown("##### Player Impact Analysis")
    st.caption("Win/loss record and scouting notes based on each player's scoring threshold")
    p_cols = st.columns(min(len(player_impacts), 3))
    for col, p in zip(p_cols, player_impacts):
        with col:
            with st.container(border=True):
                st.markdown("## 👤")
                st.markdown(f"**{p['name']}**")
                st.markdown(f"Season avg: **{p['season_avg']} pts/g**")
                if p.get("fg_pct") is not None:
                    st.markdown(f"FG%: **{p['fg_pct']}%**")
                if p.get("blurb"):
                    st.markdown("&nbsp;", unsafe_allow_html=True)
                    st.markdown(p["blurb"])
                st.markdown("&nbsp;", unsafe_allow_html=True)
                pc1, pc2 = st.columns(2)
                pc1.metric(
                    f"≥ {p['threshold']} pts",
                    p["record_above"],
                    delta=f"{p['wr_above']:.0%} WR",
                )
                pc2.metric(
                    f"< {p['threshold']} pts",
                    p["record_below"],
                    delta=f"{p['wr_below']:.0%} WR",
                    delta_color="inverse" if p["wr_below"] < p["wr_above"] else "normal",
                )
    st.markdown("&nbsp;", unsafe_allow_html=True)


# ── 3-Point Shooting ───────────────────────────────────────────────────────────

shooting = report["shooting"]
if shooting:
    st.markdown("##### 3-Point Shooting Tendency")
    with st.container(border=True):
        if shooting["pct"] >= 33:
            icon_line = "## 🟢\n**Strong threat from three**"
        elif shooting["pct"] >= 25:
            icon_line = "## 🟡\n**Moderate 3PT threat**"
        else:
            icon_line = "## 🔴\n**Weak from three — sag off**"
        st.markdown(icon_line)
        if shooting.get("blurb"):
            st.markdown(shooting["blurb"])
        st.markdown("&nbsp;", unsafe_allow_html=True)
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("3PT%", f"{shooting['pct']}%")
        tc2.metric("Attempts / Game", shooting["att_per_game"])
        tc3.metric("Season Total", f"{shooting['made']}/{shooting['att']}")
