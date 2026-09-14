"""Highland Lady Bulldogs — coach dashboard.

Layout follows the 2026 design handoff. Two rules govern how it is built:

  * Anything interactive is a native Streamlit widget (selectbox, tabs,
    dataframe), so behavior and accessibility come for free.
  * Anything purely visual is emitted as one HTML block.

The second rule is a deliberate departure from the handoff, which suggested
`st.container(border=True)` for the keys row. Those cells contain no widgets, and
reproducing the shared-rule grid through container overrides means depending on
Streamlit's internal `data-testid` DOM, which changes between releases. A single
CSS grid matches the design exactly and cannot break on upgrade. Interactive
regions still use native widgets, so the tradeoff costs nothing.

All copy generation (imperative directives, role tags, the head-to-head read)
lives in `scouting.py`. This module only arranges and escapes.
"""

import html
import re

import pandas as pd
import streamlit as st

from scouting import (
    generate_full_scouting_report,
    head_to_head_read,
    assign_role_tags,
    insight_presentation,
)
from scraper import (
    KNOWN_OPPONENTS,
    OPPONENT_METADATA,
    aggregate_player_stats,
    get_opponent_season_data,
    get_schedule,
)
from simulation import simulate_matchup

st.set_page_config(
    page_title="Highland Lady Bulldogs",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Palette ───────────────────────────────────────────────────────────────────
# The four theme colors are fixed by .streamlit/config.toml; the rest of the ramp
# is derived here so the CSS and the Streamlit theme cannot drift apart.

BG = "#f3f2f2"
INK = "#201e1d"
ACCENT = "#ec3013"
ACCENT_200 = "#fbdcd7"
ACCENT_700 = "#c2260f"
ACCENT_800 = "#8f1c0b"
N300 = "#dedbda"
N400 = "#b3afad"
N700 = "#6d6866"
N800 = "#423e3c"
DIVIDER = "#c9c6c4"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;700;800&display=swap');

html, body, [class*="st-"], button, input, select, textarea {{
    font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif;
}}

/* Every radius is 0. This single rule does most of the visual work. */
*, *::before, *::after {{ border-radius: 0 !important; }}

/* The design supplies its own header, so the Streamlit chrome goes. */
#MainMenu, footer, header[data-testid="stHeader"] {{ display: none !important; }}

.block-container {{
    max-width: 1400px;
    padding: 0 32px 80px !important;
}}

/* Controls: small uppercase labels are rendered as HTML above each collapsed
   selectbox, so only the control itself needs styling here. */
div[data-baseweb="select"] > div {{
    background: {BG};
    border: 2px solid {DIVIDER};
    box-shadow: none;
    font-size: 14px;
    font-weight: 600;
}}
div[data-baseweb="select"] > div:hover {{ border-color: {INK}; }}

/* Tabs: kill the default red pill, use a 2px ink underline instead. */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    border-bottom: 2px solid {DIVIDER};
}}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{ display: none !important; }}
.stTabs [data-baseweb="tab"] {{
    padding: 10px 18px;
    margin-bottom: -2px;
    border-bottom: 2px solid transparent;
    color: {N700};
    font-weight: 800;
    font-size: 14px;
    letter-spacing: 0.02em;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    color: {INK};
    border-bottom: 2px solid {INK};
}}

[data-testid="stDataFrame"] {{ border: 2px solid {DIVIDER}; }}

/* Streamlit attaches hover anchor-links to every heading it renders, including
   the <h4> inside our card markup. The design has no such affordance. */
[data-testid="stHeaderActionElements"] {{ display: none !important; }}
.stMarkdown h1 a.anchor-link, .stMarkdown h2 a.anchor-link,
.stMarkdown h3 a.anchor-link, .stMarkdown h4 a.anchor-link,
.stMarkdown h5 a.anchor-link, .stMarkdown h6 a.anchor-link {{ display: none !important; }}

/* Shared primitives for the HTML blocks below. */
.kicker {{
    font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
    font-weight: 600; color: {N700};
}}
.num {{ font-weight: 800; letter-spacing: -0.03em; line-height: 1; }}
.rule-2 {{ border-bottom: 2px solid {DIVIDER}; }}
.cellgrid {{
    display: grid; gap: 2px; background: {DIVIDER};
    border: 2px solid {DIVIDER};
}}
.cell {{ background: {BG}; padding: 22px; }}
.ledger-row {{
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 9px 0; border-bottom: 1px solid {N300};
}}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


def _esc(value) -> str:
    """Escape any data-derived string before it enters an HTML block."""
    return html.escape(str(value), quote=True)


def _html(markup: str) -> None:
    """Emit an HTML block.

    Leading indentation is stripped and blank lines dropped: Streamlit runs this
    through a markdown processor first, where an indented line becomes a code
    block and a blank line ends HTML mode. Newlines between tags are kept, so
    text that wraps in the source still renders with a space.
    """
    st.markdown(
        "\n".join(line.strip() for line in markup.splitlines() if line.strip()),
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────

_html(f"""
<header style="display:flex;align-items:flex-end;justify-content:space-between;gap:24px;
               padding:28px 0 14px;border-bottom:2px solid {DIVIDER};">
  <div style="display:flex;align-items:center;gap:14px;">
    <div style="width:26px;height:26px;background:{ACCENT};flex:none;"></div>
    <div style="font-weight:800;font-size:26px;letter-spacing:-0.02em;line-height:1;">
      HIGHLAND LADY BULLDOGS</div>
  </div>
  <div style="display:flex;gap:28px;font-size:11px;letter-spacing:0.1em;
              text-transform:uppercase;color:{N700};padding-bottom:3px;">
    <span>2025&ndash;26 Season</span>
    <span>Pregame Model</span>
    <span style="color:{ACCENT_700};font-weight:600;">Live data</span>
  </div>
</header>
""")


# ── Matchup controls ──────────────────────────────────────────────────────────

_SOS_LABELS = {
    1: "1 — Easiest", 2: "2 — Easy", 3: "3 — Average", 4: "4 — Hard", 5: "5 — Hardest",
}


def _control_label(text: str, accent: bool = False) -> str:
    color = ACCENT_700 if accent else INK
    return (
        f'<div style="font-size:10px;letter-spacing:0.12em;text-transform:uppercase;'
        f'font-weight:600;color:{color};margin:18px 0 4px;">{_esc(text)}</div>'
    )


c1, c2, c3, c4 = st.columns(4)
with c1:
    _html(_control_label("My team"))
    st.selectbox("My team", ["Highland"], label_visibility="collapsed")
with c2:
    _html(_control_label("Highland schedule strength"))
    hld_sos = st.selectbox(
        "Highland schedule strength", [1, 2, 3, 4, 5], index=2,
        format_func=lambda x: _SOS_LABELS[x], label_visibility="collapsed",
        key="hld_sos",
    )
with c3:
    _html(_control_label("Opponent", accent=True))
    opponent_name = st.selectbox(
        "Opponent", list(KNOWN_OPPONENTS.keys()), label_visibility="collapsed",
        key="opponent",
    )
with c4:
    _html(_control_label("Opponent schedule strength"))
    opp_sos = st.selectbox(
        "Opponent schedule strength", [1, 2, 3, 4, 5], index=2,
        format_func=lambda x: _SOS_LABELS[x], label_visibility="collapsed",
        key="opp_sos",
    )

_html(f'<div style="border-bottom:2px solid {DIVIDER};margin-top:20px;"></div>')

meta = OPPONENT_METADATA[opponent_name]


# ── Data ──────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=3600, show_spinner=False)
def load_schedule() -> pd.DataFrame:
    return get_schedule()


@st.cache_data(ttl=3600, show_spinner=False)
def load_opponent_data(team_id: int):
    return get_opponent_season_data(team_id)


with st.spinner("Loading Highland schedule..."):
    hld_schedule = load_schedule()
hld_played = hld_schedule[hld_schedule["result"].isin(["W", "L"])]

with st.spinner(f"Loading {opponent_name} season data — first load ~15 s, then cached 1 hour..."):
    opp_schedule, opp_off_df, opp_def_df = load_opponent_data(meta["team_id"])
opp_played = opp_schedule[opp_schedule["result"].isin(["W", "L"])]

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

report = generate_full_scouting_report(
    opp_schedule, opp_off_df, opp_def_df, opponent_name,
    has_turnovers=meta["has_turnovers"],
    has_fg_pct=meta["has_fg_pct"],
)
trend = report["trend"]

h2h_frame = hld_schedule[
    hld_schedule["opponent"].str.contains(opponent_name, case=False, na=False)
]
h2h = head_to_head_read(h2h_frame, sim["win_prob"], opponent_name)


# ── Verdict ───────────────────────────────────────────────────────────────────

win_pct = sim["win_prob"]
loss_pct = sim["loss_prob"]

if hld_sos != opp_sos:
    model_note = (
        f"Raw (no SOS): {sim['raw_win_prob']:.0%} &middot; "
        f"SOS-adjusted: {win_pct:.0%} "
        f"(Highland {hld_sos} vs {_esc(opponent_name)} {opp_sos})"
    )
else:
    model_note = (
        f"Highland {sim['hld_n']} games &middot; {_esc(opponent_name)} {sim['opp_n']} games "
        f"&middot; scoring distributions only"
    )


def _ledger_row(label: str, value: str, last: bool = False) -> str:
    border = "" if last else f"border-bottom:1px solid {N300};"
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
        f'padding:9px 0;{border}">'
        f'<span style="font-size:13px;color:{N700};">{_esc(label)}</span>'
        f'<span class="num" style="font-size:22px;">{value}</span></div>'
    )


if trend.get("n_games", 0) >= 4:
    recent_wr = trend.get("recent_win_rate", 0.0)
    early_wr = trend.get("early_win_rate", 0.0)
    arrow = "&uarr;" if trend.get("trending_up") else "&darr;"
    form_cell = (
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;padding:9px 0;">'
        f'<span style="font-size:13px;color:{N700};">2nd-half form</span>'
        f'<span style="text-align:right;">'
        f'<span class="num" style="font-size:22px;">{recent_wr:.0%}</span>'
        f'<span style="display:block;font-size:11px;letter-spacing:0.06em;'
        f'text-transform:uppercase;color:{ACCENT_700};font-weight:600;">'
        f'{arrow} {abs(recent_wr - early_wr):.0%} vs first half</span></span></div>'
    )
    ledger_body = (
        _ledger_row("Record", _esc(trend.get("season_record", "—")))
        + _ledger_row("Points per game", _esc(trend.get("season_ppg", "—")))
        + _ledger_row("Points allowed", _esc(trend.get("season_opp_ppg", "—")))
        + _ledger_row("Last 5", _esc(trend.get("last5_record", "—")))
        + form_cell
    )
else:
    ledger_body = (
        f'<div style="font-size:13px;color:{N700};padding:9px 0;">'
        f'Not enough completed games to summarize the season.</div>'
    )

if h2h["games"]:
    h2h_body = "".join(
        f'<div style="padding:9px 0;border-bottom:1px solid {N300};">'
        f'<div class="num" style="font-size:22px;">{g["result"]} {g["hld_score"]}&ndash;{g["opp_score"]}</div>'
        f'<div style="font-size:12px;color:{N700};">{_esc(g["date"])} &middot; {g["margin"]:+d}</div>'
        f'</div>'
        for g in h2h["games"]
    )
else:
    h2h_body = f'<div style="font-size:13px;color:{N700};padding:9px 0;">No meeting this season.</div>'

# The verdict copy is markdown from scouting.py; convert its **bold** to HTML.
# Escape first, then substitute, so the markup is the only HTML that survives.
verdict_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", _esc(h2h["verdict"]))

_html(f"""
<section style="display:grid;grid-template-columns:minmax(0,1.55fr) minmax(0,1fr) minmax(0,0.9fr);
                border-bottom:2px solid {DIVIDER};">
  <div style="padding:26px 32px 28px 0;">
    <div style="display:flex;align-items:baseline;gap:10px;font-size:11px;letter-spacing:0.12em;
                text-transform:uppercase;color:{N700};margin-bottom:18px;">
      <span style="font-weight:600;color:{INK};">Projected result</span>
      <span>{sim['n_sims']:,} Monte Carlo runs</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr auto 1fr;align-items:end;gap:16px;margin-bottom:22px;">
      <div>
        <div style="font-size:12px;letter-spacing:0.1em;text-transform:uppercase;font-weight:600;margin-bottom:2px;">Highland</div>
        <div class="num" style="font-size:96px;line-height:0.82;letter-spacing:-0.045em;">{sim['hld_projected']}</div>
        <div style="font-size:12px;color:{N700};margin-top:8px;">avg {sim['hld_mean']} &plusmn; {sim['hld_std']}</div>
      </div>
      <div class="num" style="font-size:30px;color:{N400};padding-bottom:14px;">&ndash;</div>
      <div>
        <div style="font-size:12px;letter-spacing:0.1em;text-transform:uppercase;font-weight:600;margin-bottom:2px;color:{ACCENT_700};">{_esc(opponent_name)}</div>
        <div class="num" style="font-size:96px;line-height:0.82;letter-spacing:-0.045em;color:{ACCENT};">{sim['opp_projected']}</div>
        <div style="font-size:12px;color:{N700};margin-top:8px;">avg {sim['opp_mean']} &plusmn; {sim['opp_std']}</div>
      </div>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:baseline;font-size:11px;
                letter-spacing:0.1em;text-transform:uppercase;font-weight:600;margin-bottom:6px;">
      <span>Highland win {win_pct:.0%}</span>
      <span style="color:{ACCENT_700};">{_esc(opponent_name)} win {loss_pct:.0%}</span>
    </div>
    <div style="display:flex;height:26px;width:100%;border:1px solid {DIVIDER};">
      <div style="width:{win_pct * 100:.1f}%;background:{INK};"></div>
      <div style="width:{loss_pct * 100:.1f}%;background:{ACCENT};"></div>
    </div>
    <div style="font-size:12px;color:{N700};margin-top:10px;">{model_note}</div>
  </div>
  <div style="padding:26px 32px 28px;border-left:2px solid {DIVIDER};">
    <div style="font-size:11px;letter-spacing:0.12em;text-transform:uppercase;font-weight:600;margin-bottom:16px;">{_esc(opponent_name)} this season</div>
    {ledger_body}
  </div>
  <div style="padding:26px 0 28px 32px;border-left:2px solid {DIVIDER};display:flex;flex-direction:column;">
    <div style="font-size:11px;letter-spacing:0.12em;text-transform:uppercase;font-weight:600;margin-bottom:16px;">Head to head</div>
    {h2h_body}
    <div style="margin-top:16px;font-size:13px;line-height:1.5;text-wrap:pretty;">{verdict_html}</div>
  </div>
</section>
""")


# ── Data gap strip ────────────────────────────────────────────────────────────

missing = []
if not meta["has_turnovers"]:
    missing.append("individual turnovers")
if not meta["has_fg_pct"]:
    missing.append("field goal percentages")

if missing:
    _html(f"""
    <section style="display:flex;align-items:center;gap:16px;padding:12px 0;
                    border-bottom:2px solid {DIVIDER};">
      <span style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;font-weight:700;
                   padding:4px 8px;background:{ACCENT_200};color:{ACCENT_800};flex:none;">Incomplete data</span>
      <span style="font-size:13px;">{_esc(opponent_name)} does not report {_esc(" and ".join(missing))}.
        These stats are excluded from the simulation for <em>both</em> teams so the comparison stays fair.</span>
    </section>
    """)
else:
    _html(f"""
    <section style="display:flex;align-items:center;gap:16px;padding:12px 0;
                    border-bottom:2px solid {DIVIDER};">
      <span style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;font-weight:700;
                   padding:4px 8px;background:{N300};color:{N800};flex:none;">Complete data</span>
      <span style="font-size:13px;">{_esc(opponent_name)} tracks the same categories Highland does &mdash;
        scoring, shooting percentages and turnovers all feed the comparison.</span>
    </section>
    """)


# ── Keys to the game ──────────────────────────────────────────────────────────

insights = [insight_presentation(i, opponent_name) for i in report["insights"]][:3]

_html(f"""
<section style="padding:34px 0 0;">
  <div style="display:flex;align-items:baseline;gap:10px;margin-bottom:20px;">
    <h3 style="margin:0;font-size:25px;font-weight:800;letter-spacing:-0.02em;">Keys to the game</h3>
    <span style="margin-left:auto;font-size:12px;color:{N700};">
      Thresholds derived from {_esc(opponent_name)}&rsquo;s {trend.get('n_games', 0)} completed games</span>
  </div>
</section>
""")

if insights:
    cells = []
    for idx, ins in enumerate(insights, start=1):
        if ins.get("is_synthetic"):
            footer = (
                f'<div style="border-top:2px solid {DIVIDER};padding-top:14px;margin-top:16px;">'
                f'<div class="kicker" style="margin-bottom:2px;">Coaching judgment</div>'
                f'<div class="num" style="font-size:30px;color:{N400};">No data</div></div>'
            )
            context_color = N700
        else:
            footer = (
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;'
                f'border-top:2px solid {DIVIDER};padding-top:14px;margin-top:16px;">'
                f'<div><div class="kicker" style="margin-bottom:2px;">{_esc(ins["good_label"])}</div>'
                f'<div class="num" style="font-size:30px;color:{ACCENT};">{_esc(ins["good_record"])}</div></div>'
                f'<div><div class="kicker" style="margin-bottom:2px;">{_esc(ins["bad_label"])}</div>'
                f'<div class="num" style="font-size:30px;">{_esc(ins["bad_record"])}</div></div></div>'
            )
            context_color = ACCENT_700

        cells.append(
            f'<article class="cell">'
            f'<div style="display:flex;align-items:baseline;justify-content:space-between;margin-bottom:12px;">'
            f'<span class="num" style="font-size:13px;letter-spacing:0.1em;color:{ACCENT};">{idx:02d}</span>'
            f'<span class="kicker">{_esc(ins["kicker"])}</span></div>'
            f'<h4 style="margin:0 0 4px;font-size:20px;font-weight:800;letter-spacing:-0.02em;">{_esc(ins["directive"])}</h4>'
            f'<div style="font-size:13px;font-weight:600;color:{context_color};margin-bottom:12px;">{_esc(ins["context"])}</div>'
            f'<p style="font-size:13px;line-height:1.55;color:{N800};text-wrap:pretty;margin:0;">{_esc(ins["blurb"])}</p>'
            f'{footer}</article>'
        )

    cols = f"repeat({len(cells)}, minmax(0, 1fr))"
    _html(f'<div class="cellgrid" style="grid-template-columns:{cols};">{"".join(cells)}</div>')
else:
    _html(f'<div style="font-size:13px;color:{N700};padding:8px 0 24px;">'
          f'Not enough completed games to surface meaningful win/loss patterns.</div>')


# ── Tabs ──────────────────────────────────────────────────────────────────────

_html('<div style="padding-top:34px;"></div>')
tab_personnel, tab_roster, tab_log = st.tabs(["Personnel", "Full roster", "Game log"])

with tab_personnel:
    impacts = report["player_impacts"]
    if impacts:
        _html(f'<div style="font-size:13px;color:{N700};padding:26px 0 18px;">'
              f'Win&ndash;loss splits at each scorer&rsquo;s threshold. Shut down the top line and '
              f'{_esc(opponent_name)}&rsquo;s record collapses.</div>')
        cells = []
        tags = assign_role_tags(impacts)
        for p, tag in zip(impacts, tags):
            fg = (f'<div><div class="kicker">FG%</div>'
                  f'<div class="num" style="font-size:24px;">{p["fg_pct"]}</div></div>'
                  if p.get("fg_pct") is not None else "")
            cells.append(
                f'<article class="cell">'
                f'<div style="display:flex;align-items:baseline;justify-content:space-between;'
                f'border-bottom:2px solid {DIVIDER};padding-bottom:12px;margin-bottom:14px;">'
                f'<h4 style="margin:0;font-size:20px;font-weight:800;letter-spacing:-0.02em;">{_esc(p["name"])}</h4>'
                f'<span style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;'
                f'font-weight:700;color:{ACCENT};white-space:nowrap;flex:none;'
                f'padding-left:10px;">{_esc(tag)}</span></div>'
                f'<div style="display:flex;gap:28px;margin-bottom:14px;">'
                f'<div><div class="kicker">Pts / game</div>'
                f'<div class="num" style="font-size:24px;">{p["season_avg"]}</div></div>{fg}</div>'
                f'<p style="font-size:13px;line-height:1.55;color:{N800};text-wrap:pretty;margin:0;">{_esc(p["blurb"])}</p>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;'
                f'border-top:2px solid {DIVIDER};padding-top:14px;margin-top:14px;">'
                f'<div><div class="kicker">&ge; {p["threshold"]} pts</div>'
                f'<div class="num" style="font-size:28px;color:{ACCENT};">{_esc(p["record_above"])}</div>'
                f'<div style="font-size:11px;color:{N700};">{p["wr_above"]:.0%} win rate</div></div>'
                f'<div><div class="kicker">&lt; {p["threshold"]} pts</div>'
                f'<div class="num" style="font-size:28px;">{_esc(p["record_below"])}</div>'
                f'<div style="font-size:11px;color:{N700};">{p["wr_below"]:.0%} win rate</div></div></div>'
                f'</article>'
            )
        cols = f"repeat({len(cells)}, minmax(0, 1fr))"
        _html(f'<div class="cellgrid" style="grid-template-columns:{cols};">{"".join(cells)}</div>')
    else:
        _html(f'<div style="font-size:13px;color:{N700};padding:26px 0;">'
              f'Not enough per-player data to build impact splits for {_esc(opponent_name)}.</div>')

with tab_roster:
    player_stats = aggregate_player_stats(opp_off_df, opp_def_df)
    if not player_stats.empty:
        # Per-game rates first; season totals other than points are dropped, since
        # the per-game rate is what a coach scans for.
        wanted = ["GP", "PPG", "Pts", "FG%", "3FG%", "FT%", "RPG", "APG", "SPG"]
        cols = [c for c in wanted if c in player_stats.columns]
        table = player_stats[cols].head(10).reset_index()
        table = table.rename(columns={table.columns[0]: "Player"})

        one_dp = {"PPG", "FG%", "3FG%", "FT%", "RPG", "APG", "SPG"}
        column_config = {
            c: st.column_config.NumberColumn(c, format="%.1f")
            for c in cols if c in one_dp
        }
        for c in ("GP", "Pts"):
            if c in cols:
                column_config[c] = st.column_config.NumberColumn(c, format="%d")

        _html('<div style="padding-top:26px;"></div>')
        st.dataframe(table, use_container_width=True, hide_index=True,
                     column_config=column_config)
    else:
        _html(f'<div style="font-size:13px;color:{N700};padding:26px 0;">'
              f'No box score data available for {_esc(opponent_name)}.</div>')

with tab_log:
    recent = opp_played.tail(5).iloc[::-1]
    if not recent.empty:
        rows = []
        for i, (_, g) in enumerate(recent.iterrows()):
            last = i == len(recent) - 1
            border = "" if last else f"border-bottom:1px solid {N300};"
            color = f"color:{ACCENT};" if g["result"] == "L" else ""
            rows.append(
                f'<div style="display:flex;justify-content:space-between;padding:10px 0;{border}font-size:14px;">'
                f'<span>{_esc(g["opponent"])}</span>'
                f'<span style="font-weight:700;{color}">{g["result"]} '
                f'{int(g["hld_score"])}&ndash;{int(g["opp_score"])}</span></div>'
            )
        results_body = "".join(rows)
    else:
        results_body = f'<div style="font-size:13px;color:{N700};">No completed games.</div>'

    excluded = ", ".join(m.split()[-1].title() for m in missing) if missing else "None"
    sos_note = "None (3 vs 3)" if hld_sos == opp_sos else f"{hld_sos} vs {opp_sos}"

    def _input_row(label: str, value: str, accent: bool = False, last: bool = False) -> str:
        border = "" if last else f"border-bottom:1px solid {N300};"
        color = f"color:{ACCENT};" if accent else ""
        return (
            f'<div style="display:flex;justify-content:space-between;padding:10px 0;{border}font-size:14px;">'
            f'<span>{_esc(label)}</span>'
            f'<span style="font-weight:700;{color}">{_esc(value)}</span></div>'
        )

    _html(f"""
    <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:40px;padding-top:26px;">
      <div>
        <div style="font-size:11px;letter-spacing:0.12em;text-transform:uppercase;font-weight:600;margin-bottom:12px;">
          {_esc(opponent_name)} &mdash; recent results</div>
        {results_body}
      </div>
      <div>
        <div style="font-size:11px;letter-spacing:0.12em;text-transform:uppercase;font-weight:600;margin-bottom:12px;">
          Model inputs</div>
        {_input_row("Simulations", f"{sim['n_sims']:,}")}
        {_input_row("Highland sample", f"{sim['hld_n']} games")}
        {_input_row(f"{opponent_name} sample", f"{sim['opp_n']} games")}
        {_input_row("SOS adjustment", sos_note)}
        {_input_row("Excluded categories", excluded, accent=bool(missing), last=True)}
      </div>
    </div>
    """)
