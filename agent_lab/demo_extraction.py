"""Exercise 4 — drives the extraction pipeline through every outcome.

Run:  .venv/Scripts/python.exe -m agent_lab.demo_extraction

Part 1 unit-tests the invariant checker directly — if the validator is wrong,
everything built on it is theatre. Part 2 runs the full extract/repair/reconcile
loop against a scripted model.
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent_lab.extraction import (
    GameReport,
    PlayerLine,
    Status,
    check_invariants,
    extract_batch,
    extract_game,
    reconcile,
)
from agent_lab.model import (
    MODEL_SONNET,
    PermanentModelError,
    ScriptedClient,
    ScriptedTurn,
)


def line(name, fgm, fga, tpm, tpa, ftm, fta, pts=None):
    """Build a player line; points default to the correct value."""
    return PlayerLine(
        name=name, points=pts if pts is not None else 2 * fgm + tpm + ftm,
        fg_made=fgm, fg_att=fga, three_made=tpm, three_att=tpa,
        ft_made=ftm, ft_att=fta,
    )


# A real-shaped Highland win: 49-36, five scorers summing exactly to 49.
CLEAN_PLAYERS = [
    line("Kaufmann", 5, 11, 1, 3, 2, 2),   # 13
    line("Brave", 6, 13, 0, 1, 0, 0),      # 12
    line("Schwarz", 4, 9, 2, 5, 1, 2),     # 11
    line("Hemann", 3, 7, 1, 2, 2, 4),      # 9
    line("Wilson", 2, 5, 0, 0, 0, 0),      # 4
]
CLEAN = GameReport(opponent="Triad", date="Jan 14", highland_score=49,
                   opponent_score=36, result="W", players=CLEAN_PLAYERS)

# Same game, transcribed badly: Kaufmann's points do not follow from her makes,
# and Brave has more makes than attempts.
BROKEN = GameReport(
    opponent="Triad", date="Jan 14", highland_score=49, opponent_score=36, result="W",
    players=[
        line("Kaufmann", 5, 11, 1, 3, 2, 2, pts=15),
        line("Brave", 6, 4, 0, 1, 0, 0),
        *CLEAN_PLAYERS[2:],
    ],
)

DOC = """Beat Triad 49-36 on the 14th.
Kaufmann 13 (5-11, 1-3 from deep, 2-2 FT), Brave 12 on 6-13,
Schwarz 11 (4-9, 2-5 threes, 1-2 line), Hemann 9, Wilson 4."""


def part1_validator() -> None:
    print("-" * 78)
    print("Part 1 — does the invariant checker actually catch bad arithmetic?")
    print("-" * 78)

    cases = [
        ("a consistent record passes", CLEAN, 0),
        ("points not derivable from makes", GameReport(
            opponent="Triad", date="Jan 14", highland_score=15, opponent_score=36,
            result="L", players=[line("X", 5, 11, 1, 3, 2, 2, pts=15)]), 1),
        ("more makes than attempts", GameReport(
            opponent="Triad", date="Jan 14", highland_score=12, opponent_score=36,
            result="L", players=[line("X", 6, 4, 0, 1, 0, 0)]), 1),
        ("threes exceed field goals", GameReport(
            opponent="Triad", date="Jan 14", highland_score=11, opponent_score=36,
            result="L", players=[line("X", 2, 9, 3, 5, 1, 2)]), 2),
        ("player points do not sum to team score", GameReport(
            opponent="Triad", date="Jan 14", highland_score=60, opponent_score=36,
            result="W", players=CLEAN_PLAYERS), 1),
        ("result contradicts the score", GameReport(
            opponent="Triad", date="Jan 14", highland_score=36, opponent_score=49,
            result="W", players=[]), 1),
    ]

    for name, report, expected in cases:
        found = check_invariants(report)
        ok = len(found) == expected
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {len(found)} violation(s) (expected {expected})")
        for v in found:
            print(f"       - {v}")
        assert ok, f"{name}: expected {expected}, got {found}"

    # Reconciliation is a separate layer: internally perfect, wrong game.
    mism = reconcile(CLEAN, {"highland_score": 52, "opponent_score": 36, "opponent": "Triad"})
    print(f"ok   reconciliation catches a wrong-game record: {mism}")
    assert len(mism) == 1
    print()


def part2_pipeline() -> None:
    print("-" * 78)
    print("Part 2 — extract / repair / reconcile / quarantine")
    print("-" * 78)

    # 1. Clean on the first pass.
    client = ScriptedClient({MODEL_SONNET: [ScriptedTurn(parsed=CLEAN)]})
    out = extract_game("g1", DOC, client=client)
    print(f"ok   first-pass clean      -> {out.status.value}, {out.repair_passes} repair pass(es)")
    assert out.status is Status.CLEAN and out.usable and len(client.calls) == 1

    # 2. Broken, then fixed when told exactly what was wrong.
    client = ScriptedClient({MODEL_SONNET: [
        ScriptedTurn(parsed=BROKEN), ScriptedTurn(parsed=CLEAN)]})
    out = extract_game("g2", DOC, client=client)
    print(f"ok   repaired after 1 pass -> {out.status.value}, {out.repair_passes} repair pass(es)")
    assert out.status is Status.REPAIRED and out.usable
    # The repair prompt must contain the specific violations, not a generic retry.
    repair_prompt = client.calls[1].messages[-1]["content"]
    assert "points 15" in repair_prompt and "exceeds fg_att" in repair_prompt
    print("     repair prompt cited the actual violations, not a blind retry")

    # 3. Never converges -> quarantined with reasons, nothing invented.
    client = ScriptedClient({MODEL_SONNET: [ScriptedTurn(parsed=BROKEN) for _ in range(3)]})
    out = extract_game("g3", DOC, client=client)
    print(f"ok   never converges       -> {out.status.value} after {len(client.calls)} calls")
    assert out.status is Status.QUARANTINED and out.report is None and not out.usable
    assert out.violations, "a quarantined record must carry its reasons"

    # 4. Internally valid but disagrees with the scraped box score.
    client = ScriptedClient({MODEL_SONNET: [ScriptedTurn(parsed=CLEAN)]})
    out = extract_game("g4", DOC, client=client,
                       truth={"highland_score": 52, "opponent_score": 36, "opponent": "Triad"})
    print(f"ok   disagrees with truth  -> {out.status.value} (usable={out.usable})")
    assert out.status is Status.MISMATCHED and not out.usable
    print(f"     {out.violations[0]}")

    # 5. Model failure is contained.
    client = ScriptedClient({MODEL_SONNET: [ScriptedTurn(raises=PermanentModelError("400"))]})
    out = extract_game("g5", DOC, client=client)
    print(f"ok   model error           -> {out.status.value}")
    assert out.status is Status.QUARANTINED

    print()


def part3_batch() -> None:
    print("-" * 78)
    print("Part 3 — batch: one bad note must not sink the season")
    print("-" * 78)

    client = ScriptedClient({MODEL_SONNET: [
        ScriptedTurn(parsed=CLEAN),                       # jan14 clean
        ScriptedTurn(parsed=BROKEN), ScriptedTurn(parsed=CLEAN),  # jan17 repaired
        ScriptedTurn(parsed=BROKEN), ScriptedTurn(parsed=BROKEN),
        ScriptedTurn(parsed=BROKEN),                      # jan21 quarantined
        ScriptedTurn(parsed=CLEAN),                       # jan24 mismatched vs truth
    ]})
    docs = [("jan14", DOC), ("jan17", DOC), ("jan21", DOC), ("jan24", DOC)]
    batch = extract_batch(
        docs, client=client,
        truth_by_doc={"jan24": {"highland_score": 52, "opponent_score": 36}},
    )
    print(batch.render())
    assert len(batch.usable) == 2
    assert len(batch.quarantined) == 1
    assert len(batch.mismatched) == 1
    print()


def main() -> None:
    print("=" * 78)
    print("Exercise 4 — structured extraction pipeline")
    print("=" * 78, "\n")
    part1_validator()
    part2_pipeline()
    part3_batch()
    print("=" * 78)
    print("extraction pipeline behaved as designed")
    print("=" * 78)


if __name__ == "__main__":
    main()
