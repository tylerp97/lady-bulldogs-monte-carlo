"""Exercise 3 — the debug half.

Run:  .venv/Scripts/python.exe -m agent_lab.debug_research

Each scenario drives `research_opponent` into a specific failure mode and checks
that the pipeline degrades the way the design claims it does. Scenarios that
FAIL here are real defects in `research.py`, not in the harness — that is the
point of the exercise.
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent_lab.model import (
    ModelRequest,
    PermanentModelError,
    RoutingScriptedClient,
    ScriptedTurn,
    TransientModelError,
)
from agent_lab.research import (
    DEFAULT_ANGLES,
    Angle,
    Finding,
    ScoutingBrief,
    WorkerReport,
    research_opponent,
)

# A Triad-shaped bundle: scoring and shooting present, turnovers absent — exactly
# the gap `OPPONENT_METADATA["Triad"]["has_turnovers"] = False` describes.
BUNDLE = {
    "scoring_ppg": 51.0,
    "scoring_last5_ppg": 47.4,
    "shooting_3pt_pct": 31.2,
    "shooting_3pt_att_pg": 14.0,
    "allowed_ppg": 43.8,
    "allowed_last5_ppg": 46.0,
    "player_top_scorer": "M. Reiher",
    "player_top_scorer_ppg": 18.3,
    "player_second_ppg": 11.1,
}

NO_TURNOVER_BUNDLE = {k: v for k, v in BUNDLE.items() if not k.startswith("turnover_")}


def router(request: ModelRequest) -> str:
    """Map a request to a script tag by reading its system prompt."""
    system = request.system
    if system.startswith("You are the head scout"):
        return "synthesize"
    for angle in DEFAULT_ANGLES:
        if f"a {angle.key} specialist" in system:
            return angle.key
    return "unknown"


def report(angle: str, claim: str, severity: str = "high", conf: float = 0.8) -> WorkerReport:
    return WorkerReport(
        angle=angle,
        findings=[Finding(claim=claim, evidence_keys=[f"{angle}_key"],
                          confidence=conf, severity=severity)],
    )


BRIEF = ScoutingBrief(
    headline="Triad scores in bunches but gives up points late.",
    priorities=["Contain Reiher early", "Push pace in the 4th"],
    contradictions=[],
    coverage_note="",
)

OFFENSE = report("offense", "Averages 51.0, dips to 47.4 over the last five.")
DEFENSE = report("defense", "Allows 43.8 but 46.0 recently — trending the wrong way.")
PERSONNEL = report("personnel", "Reiher at 18.3 ppg is the engine.")


def run(name, script, bundle=BUNDLE, angles=DEFAULT_ANGLES, **kwargs):
    client = RoutingScriptedClient(script, router)
    try:
        result = research_opponent("Triad", bundle, client=client, angles=angles, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the harness must survive to report
        print(f"CRASH {name}")
        print(f"      {type(exc).__name__}: {exc}")
        print()
        return None, client
    print(f"ok    {name}")
    print(f"      {result.summary()}")
    print(f"      tags called: {client.tags_called()}")
    if result.brief:
        print(f"      headline: {result.brief.headline}")
        if result.brief.contradictions:
            print(f"      contradictions: {result.brief.contradictions}")
        if result.brief.coverage_note:
            print(f"      coverage: {result.brief.coverage_note}")
    else:
        print("      brief: NONE")
    for span in result.trace.failures:
        print(f"      failed span: {span.name} — {span.error}")
    print()
    return result, client


def main() -> None:
    print("=" * 78)
    print("Exercise 3 — research pipeline under adversarial conditions")
    print("=" * 78, "\n")

    # 1. Everything works.
    result, client = run(
        "happy path: 3 workers + synthesis",
        {"offense": [ScriptedTurn(parsed=OFFENSE)],
         "defense": [ScriptedTurn(parsed=DEFENSE)],
         "personnel": [ScriptedTurn(parsed=PERSONNEL)],
         "synthesize": [ScriptedTurn(parsed=BRIEF)]},
    )
    assert result and result.brief and not result.degraded
    assert len(result.reports) == 3

    # 2. One worker dies. The brief must still ship, flagged degraded.
    result, client = run(
        "partial failure: defense worker times out",
        {"offense": [ScriptedTurn(parsed=OFFENSE)],
         "defense": [ScriptedTurn(raises=TransientModelError("timeout"))],
         "personnel": [ScriptedTurn(parsed=PERSONNEL)],
         "synthesize": [ScriptedTurn(parsed=BRIEF)]},
    )
    assert result and result.brief is not None, "one dead worker must not lose the brief"
    assert result.degraded, "losing an angle must be reported as degraded"
    assert len(result.reports) == 2

    # 3. Empty slice: no model call should be made for that angle.
    result, client = run(
        "empty slice: no turnover data -> defense angle has partial slice",
        {"offense": [ScriptedTurn(parsed=OFFENSE)],
         "defense": [ScriptedTurn(parsed=DEFENSE)],
         "personnel": [ScriptedTurn(parsed=PERSONNEL)],
         "synthesize": [ScriptedTurn(parsed=BRIEF)]},
        bundle=NO_TURNOVER_BUNDLE,
    )

    # 3b. A truly empty slice for an angle.
    only_offense = {k: v for k, v in BUNDLE.items() if k.startswith("scoring_")}
    result, client = run(
        "empty slice: personnel + defense slices empty -> zero model calls for them",
        {"offense": [ScriptedTurn(parsed=OFFENSE)],
         "synthesize": [ScriptedTurn(parsed=BRIEF)]},
        bundle=only_offense,
    )
    assert result is not None
    assert "defense" not in client.tags_called(), "empty slice must not call a model"
    assert "personnel" not in client.tags_called(), "empty slice must not call a model"

    # 4. All workers fail -> no brief, nothing invented.
    result, client = run(
        "total failure: every worker dies",
        {"offense": [ScriptedTurn(raises=TransientModelError("timeout"))],
         "defense": [ScriptedTurn(raises=TransientModelError("timeout"))],
         "personnel": [ScriptedTurn(raises=TransientModelError("timeout"))],
         "synthesize": [ScriptedTurn(parsed=BRIEF)]},
    )
    assert result and result.brief is None, "must not synthesize a brief from nothing"
    assert "synthesize" not in client.tags_called()

    # 5. Synthesizer itself fails — worker output must survive for a human.
    result, client = run(
        "synthesizer fails: worker reports must be retained",
        {"offense": [ScriptedTurn(parsed=OFFENSE)],
         "defense": [ScriptedTurn(parsed=DEFENSE)],
         "personnel": [ScriptedTurn(parsed=PERSONNEL)],
         "synthesize": [ScriptedTurn(raises=PermanentModelError("400"))]},
    )
    assert result and result.brief is None
    assert len(result.reports) == 3, "raw research must not be thrown away"

    # 6. A worker returns the wrong shape.
    result, client = run(
        "malformed worker output: personnel returns no structured output",
        {"offense": [ScriptedTurn(parsed=OFFENSE)],
         "defense": [ScriptedTurn(parsed=DEFENSE)],
         "personnel": [ScriptedTurn(text="here you go", parsed=None)],
         "synthesize": [ScriptedTurn(parsed=BRIEF)]},
    )
    assert result and result.brief is not None
    assert result.degraded

    # 7. Determinism: report order must not depend on which thread finishes first.
    orders = set()
    for _ in range(12):
        client = RoutingScriptedClient(
            {"offense": [ScriptedTurn(parsed=OFFENSE)],
             "defense": [ScriptedTurn(parsed=DEFENSE)],
             "personnel": [ScriptedTurn(parsed=PERSONNEL)],
             "synthesize": [ScriptedTurn(parsed=BRIEF)]},
            router,
        )
        res = research_opponent("Triad", BUNDLE, client=client, angles=DEFAULT_ANGLES)
        orders.add(tuple(r.angle for r in res.reports))
    print(f"ok    determinism: {len(orders)} distinct report order(s) over 12 runs -> {orders}")
    assert len(orders) == 1, f"report order is nondeterministic: {orders}"
    print()

    # 8. A worker raises something that is NOT a ModelError.
    def exploding_slice(_bundle):
        raise ValueError("slice_fn bug: unexpected column layout")

    angles = (
        DEFAULT_ANGLES[0],
        Angle("defense", DEFAULT_ANGLES[1].model, "medium", "q", exploding_slice),
        DEFAULT_ANGLES[2],
    )
    result, client = run(
        "non-ModelError in a worker: does one bad slice_fn kill the pipeline?",
        {"offense": [ScriptedTurn(parsed=OFFENSE)],
         "personnel": [ScriptedTurn(parsed=PERSONNEL)],
         "synthesize": [ScriptedTurn(parsed=BRIEF)]},
        angles=angles,
    )
    assert result is not None, "an unexpected worker exception must not crash the pipeline"
    assert result.brief is not None, "the other two angles should still produce a brief"

    print("=" * 78)
    print("all research-pipeline scenarios behaved as designed")
    print("=" * 78)


if __name__ == "__main__":
    main()
