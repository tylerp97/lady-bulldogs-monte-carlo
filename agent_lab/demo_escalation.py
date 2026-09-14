"""Drives every branch of the exercise-2 escalation ladder offline.

Run:  .venv/Scripts/python.exe -m agent_lab.demo_escalation

Each scenario scripts what each tier returns, then asserts on the path the ladder
took. No network, no API key, no cost — the point is that the *control flow* is
testable independently of model quality.
"""

from __future__ import annotations

import sys

# Windows consoles default to cp1252 and mangle the em-dashes in the answers.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent_lab.escalation import (
    CitedStat,
    CoachAnswer,
    Grounder,
    Policy,
    Reason,
    ask,
)
from agent_lab.model import (
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
    PermanentModelError,
    RefusalError,
    ScriptedClient,
    ScriptedTurn,
    TransientModelError,
)

# Ground truth the Grounder checks against — the real shape of Highland's season.
FACTS = {
    "highland_games_played": 24.0,
    "highland_wins": 15.0,
    "highland_losses": 9.0,
    "highland_ppg": 45.5,
    "highland_opp_ppg": 41.2,
    "highland_best_margin": 28.0,
    "highland_worst_margin": -19.0,
}
GROUNDER = Grounder(FACTS)

QUESTION = "How are we scoring compared to what we give up?"

NO_SLEEP = lambda _seconds: None  # noqa: E731 - keeps retry tests instant


def answer(text: str, confidence: float, cites: dict[str, float], needs_human: bool = False):
    return CoachAnswer(
        answer=text,
        confidence=confidence,
        stats_cited=[CitedStat(name=k, value=v) for k, v in cites.items()],
        needs_human=needs_human,
    )


GOOD = answer("We average 45.5 and allow 41.2 — a +4.3 margin.", 0.92,
              {"highland_ppg": 45.5, "highland_opp_ppg": 41.2})
HALLUCINATED = answer("We average 61.2 and allow 41.2.", 0.95,
                      {"highland_ppg": 61.2, "highland_opp_ppg": 41.2})
UNSURE = answer("Possibly around 45, but I am not certain.", 0.40,
                {"highland_ppg": 45.5})


def scenario(name: str, script, question=QUESTION, policy=None, expect=None):
    client = ScriptedClient(script)
    res = ask(question, client=client, grounder=GROUNDER, policy=policy, sleep=NO_SLEEP)
    status = "ok " if expect is None or res.reason is expect else "FAIL"
    print(f"{status} {name}")
    print(f"      {res.summary()}")
    print(f"      model calls made: {len(client.calls)}")
    if res.answer:
        print(f"      answer: {res.answer.answer}")
    for att in res.attempts:
        if att.detail:
            print(f"      - {att.model}: {att.detail}")
    print()
    assert expect is None or res.reason is expect, f"{name}: expected {expect}, got {res.reason}"
    return res, client


def main() -> None:
    print("=" * 78)
    print("Exercise 2 — escalation ladder, all branches")
    print("=" * 78, "\n")

    # 1. Cheap tier nails it. No escalation, one call.
    res, client = scenario(
        "happy path: Haiku confident + grounded",
        {MODEL_HAIKU: [ScriptedTurn(parsed=GOOD)]},
        expect=Reason.OK,
    )
    assert len(client.calls) == 1
    assert res.attempts[-1].model == MODEL_HAIKU

    # 2. The case self-reported confidence cannot catch: high confidence, wrong number.
    res, client = scenario(
        "hallucination: Haiku is confidently wrong -> grounding catches it",
        {MODEL_HAIKU: [ScriptedTurn(parsed=HALLUCINATED)],
         MODEL_SONNET: [ScriptedTurn(parsed=GOOD)]},
        expect=Reason.OK,
    )
    assert res.attempts[0].reason is Reason.UNGROUNDED
    assert res.attempts[-1].model == MODEL_SONNET

    # 3. Genuine uncertainty climbs the whole ladder.
    scenario(
        "low confidence: Haiku -> Sonnet -> Opus",
        {MODEL_HAIKU: [ScriptedTurn(parsed=UNSURE)],
         MODEL_SONNET: [ScriptedTurn(parsed=UNSURE)],
         MODEL_OPUS: [ScriptedTurn(parsed=GOOD)]},
        expect=Reason.OK,
    )

    # 4. Sensitive question never reaches a model at all.
    res, client = scenario(
        "policy gate: injury question -> human, zero model calls",
        {MODEL_HAIKU: [ScriptedTurn(parsed=GOOD)]},
        question="Is Ella cleared to play after her concussion?",
        expect=Reason.SENSITIVE_TOPIC,
    )
    assert len(client.calls) == 0, "a sensitive question must not reach the API"
    assert res.escalated_to_human

    # 5. Rate limiting is retried in place, then falls through to the next tier.
    res, client = scenario(
        "transient: Haiku rate-limited 3x -> Sonnet answers",
        {MODEL_HAIKU: [ScriptedTurn(raises=TransientModelError("429")) for _ in range(3)],
         MODEL_SONNET: [ScriptedTurn(parsed=GOOD)]},
        expect=Reason.OK,
    )
    assert len([c for c in client.calls if c.model == MODEL_HAIKU]) == 3, "should retry in place"

    # 6. A refusal is not an escalation trigger — another model will refuse too.
    res, client = scenario(
        "refusal: stops immediately, does not climb the ladder",
        {MODEL_HAIKU: [ScriptedTurn(raises=RefusalError("declined", category="cyber"))],
         MODEL_SONNET: [ScriptedTurn(parsed=GOOD)]},
        expect=Reason.REFUSED,
    )
    assert len(client.calls) == 1, "refusal must not escalate"

    # 7. A malformed request fails fast rather than spending on two more tiers.
    res, client = scenario(
        "permanent error: fails fast",
        {MODEL_HAIKU: [ScriptedTurn(raises=PermanentModelError("400 bad request"))],
         MODEL_SONNET: [ScriptedTurn(parsed=GOOD)]},
        expect=Reason.PERMANENT_ERROR,
    )
    assert len(client.calls) == 1

    # 8. Budget ceiling stops the ladder before the expensive tier.
    tiny_budget = Policy(budget_usd=0.0005)
    res, client = scenario(
        "budget: ceiling hit -> human instead of Opus",
        {MODEL_HAIKU: [ScriptedTurn(parsed=UNSURE)],
         MODEL_SONNET: [ScriptedTurn(parsed=GOOD)]},
        policy=tiny_budget,
        expect=Reason.BUDGET_EXCEEDED,
    )
    assert res.escalated_to_human

    # 9. Every tier hallucinates -> nobody gets a wrong answer.
    res, client = scenario(
        "exhausted: all three tiers ungrounded -> human",
        {MODEL_HAIKU: [ScriptedTurn(parsed=HALLUCINATED)],
         MODEL_SONNET: [ScriptedTurn(parsed=HALLUCINATED)],
         MODEL_OPUS: [ScriptedTurn(parsed=HALLUCINATED)]},
        expect=Reason.TIER_EXHAUSTED,
    )
    assert res.answer is None, "never return an ungrounded answer"
    assert len(client.calls) == 3

    # 10. Schema failure: no parsed output at all.
    scenario(
        "schema: unparseable at Haiku -> Sonnet recovers",
        {MODEL_HAIKU: [ScriptedTurn(text="not json", parsed=None)],
         MODEL_SONNET: [ScriptedTurn(parsed=GOOD)]},
        expect=Reason.OK,
    )

    print("=" * 78)
    print("all escalation branches behaved as designed")
    print("=" * 78)


if __name__ == "__main__":
    main()
