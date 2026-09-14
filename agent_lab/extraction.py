"""Exercise 4 — a structured extraction pipeline.

Turns unstructured game notes (a coach's texted summary, a pasted box score, a
scorebook photo transcription) into validated `GameReport` records.

    extract -> validate -> repair (bounded) -> reconcile -> accept | quarantine

The design principle: **a schema is not a validator.** Pydantic guarantees the
model returned an integer where an integer belongs. It says nothing about whether
that integer is *possible*. A player with 4 made field goals on 2 attempts
satisfies the schema perfectly and is nonsense.

So validation runs in three widening layers:

  1. **Schema** — types and ranges. Pydantic, free.
  2. **Invariants** — basketball arithmetic that must hold within a record.
     `points == 2*fg_made + three_made + ft_made` is an identity, not a heuristic;
     a violation is proof of a bad extraction with no judgment call involved.
  3. **Reconciliation** — agreement with the scraped box score, when one exists.
     This is the only layer that can catch a record that is internally perfect
     but describes the wrong game.

Failures at layer 2 are *repairable*: the violations name exactly what is wrong,
and feeding them back gives the model something specific to fix. That is the
difference between a repair loop and a retry loop — a plain retry resends the
same prompt and usually reproduces the same error.

Anything still failing after the repair budget is **quarantined with reasons**,
never silently dropped and never silently accepted.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Iterable

from pydantic import BaseModel, Field

from agent_lab.model import (
    MODEL_SONNET,
    ModelClient,
    ModelError,
    ModelRequest,
)


# ── Target schema ─────────────────────────────────────────────────────────────


class PlayerLine(BaseModel):
    """One player's line. Every counting stat is non-negative by construction."""

    name: str
    jersey: int | None = Field(default=None, ge=0, le=99)
    points: int = Field(ge=0)
    fg_made: int = Field(ge=0)
    fg_att: int = Field(ge=0)
    three_made: int = Field(ge=0)
    three_att: int = Field(ge=0)
    ft_made: int = Field(ge=0)
    ft_att: int = Field(ge=0)
    rebounds: int = Field(default=0, ge=0)
    assists: int = Field(default=0, ge=0)
    steals: int = Field(default=0, ge=0)
    turnovers: int = Field(default=0, ge=0)


class GameReport(BaseModel):
    opponent: str
    date: str = Field(description="As written in the source; do not reformat")
    highland_score: int = Field(ge=0)
    opponent_score: int = Field(ge=0)
    result: str = Field(description="W or L")
    players: list[PlayerLine] = Field(default_factory=list)


# ── Layer 2: invariants ───────────────────────────────────────────────────────


def check_invariants(report: GameReport) -> list[str]:
    """Return every arithmetic contradiction in the record. Empty means consistent.

    These are identities of the sport, not tunable thresholds. Each message names
    the player and the exact numbers so a repair pass has something actionable.
    """
    problems: list[str] = []

    for p in report.players:
        if p.fg_made > p.fg_att:
            problems.append(f"{p.name}: fg_made {p.fg_made} exceeds fg_att {p.fg_att}")
        if p.three_made > p.three_att:
            problems.append(
                f"{p.name}: three_made {p.three_made} exceeds three_att {p.three_att}"
            )
        if p.ft_made > p.ft_att:
            problems.append(f"{p.name}: ft_made {p.ft_made} exceeds ft_att {p.ft_att}")
        # A three-pointer is a field goal — makes and attempts are both subsets.
        if p.three_made > p.fg_made:
            problems.append(
                f"{p.name}: three_made {p.three_made} exceeds fg_made {p.fg_made} "
                f"(threes are field goals)"
            )
        if p.three_att > p.fg_att:
            problems.append(
                f"{p.name}: three_att {p.three_att} exceeds fg_att {p.fg_att} "
                f"(threes are field goals)"
            )
        # The scoring identity: every point comes from a 2, a 3, or a free throw.
        expected = 2 * p.fg_made + p.three_made + p.ft_made
        if p.points != expected:
            problems.append(
                f"{p.name}: points {p.points} but 2*fg_made({p.fg_made}) + "
                f"three_made({p.three_made}) + ft_made({p.ft_made}) = {expected}"
            )

    if report.players:
        total = sum(p.points for p in report.players)
        if total != report.highland_score:
            problems.append(
                f"team: highland_score {report.highland_score} but player points sum to {total}"
            )

    if report.result not in ("W", "L"):
        problems.append(f"team: result {report.result!r} must be 'W' or 'L'")
    elif report.result == "W" and report.highland_score <= report.opponent_score:
        problems.append(
            f"team: result 'W' but score {report.highland_score}-{report.opponent_score}"
        )
    elif report.result == "L" and report.highland_score >= report.opponent_score:
        problems.append(
            f"team: result 'L' but score {report.highland_score}-{report.opponent_score}"
        )

    return problems


# ── Layer 3: reconciliation ───────────────────────────────────────────────────


def reconcile(report: GameReport, truth: dict[str, Any] | None) -> list[str]:
    """Compare an internally-consistent record against the scraped box score.

    A record can satisfy every invariant and still be wrong — transcribed from the
    wrong game, or from a scorebook that disagrees with the official sheet. Only
    an external source catches that, so a mismatch here is reported rather than
    repaired: the extraction may be faithful to a document that is itself wrong.
    """
    if not truth:
        return []

    problems: list[str] = []
    for field in ("highland_score", "opponent_score"):
        expected = truth.get(field)
        if expected is not None and getattr(report, field) != expected:
            problems.append(
                f"{field}: extracted {getattr(report, field)}, box score says {expected}"
            )

    expected_opp = truth.get("opponent")
    if expected_opp and expected_opp.lower() not in report.opponent.lower():
        problems.append(f"opponent: extracted {report.opponent!r}, box score says {expected_opp!r}")

    return problems


# ── Outcome ───────────────────────────────────────────────────────────────────


class Status(str, enum.Enum):
    CLEAN = "clean"  # valid on the first pass
    REPAIRED = "repaired"  # valid after one or more repair passes
    MISMATCHED = "mismatched"  # internally valid, disagrees with the box score
    QUARANTINED = "quarantined"  # never reached a valid state


@dataclasses.dataclass
class ExtractionOutcome:
    doc_id: str
    status: Status
    report: GameReport | None
    violations: list[str] = dataclasses.field(default_factory=list)
    repair_passes: int = 0
    cost_usd: float = 0.0
    error: str = ""

    @property
    def usable(self) -> bool:
        """Safe to load downstream. A mismatch is surfaced, not trusted."""
        return self.status in (Status.CLEAN, Status.REPAIRED)


# ── Extraction ────────────────────────────────────────────────────────────────

_SYSTEM = """You extract structured box score data from unstructured notes written by a \
high school girls basketball coaching staff.

Rules:
- Transcribe only what the document states. Never infer or fill in a stat that is absent; \
use 0 only when the document says zero.
- three_made and three_att are SUBSETS of fg_made and fg_att — a made three is also a made \
field goal. Count it in both.
- points must equal 2*fg_made + three_made + ft_made.
- Keep the date exactly as written in the source."""

_REPAIR_TEMPLATE = """Your previous extraction violated these constraints:

{violations}

Re-extract from the same document, correcting those specific problems. Do not change \
values that were not implicated. If the document genuinely does not support a consistent \
record, prefer omitting a player over inventing numbers that balance."""


def extract_game(
    doc_id: str,
    document: str,
    *,
    client: ModelClient,
    truth: dict[str, Any] | None = None,
    model: str = MODEL_SONNET,
    max_repairs: int = 2,
) -> ExtractionOutcome:
    """Extract one record, repairing invariant violations up to `max_repairs` times."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": f"Extract the box score from this note:\n\n{document}"}
    ]
    cost = 0.0
    violations: list[str] = []

    for attempt in range(max_repairs + 1):
        request = ModelRequest(
            model=model,
            system=_SYSTEM,
            messages=messages,
            effort="medium",
            output_format=GameReport,
            max_tokens=4096,
        )
        try:
            result = client.complete(request)
        except ModelError as exc:
            return ExtractionOutcome(doc_id, Status.QUARANTINED, None, violations,
                                     attempt, cost, error=str(exc))

        cost += result.cost_usd
        report = result.parsed

        if not isinstance(report, GameReport):
            violations = ["structured output missing or failed schema validation"]
            # Nothing specific to feed back, so the next pass is a plain retry.
            continue

        violations = check_invariants(report)
        if not violations:
            mismatches = reconcile(report, truth)
            if mismatches:
                return ExtractionOutcome(doc_id, Status.MISMATCHED, report, mismatches,
                                         attempt, cost)
            status = Status.CLEAN if attempt == 0 else Status.REPAIRED
            return ExtractionOutcome(doc_id, status, report, [], attempt, cost)

        if attempt == max_repairs:
            break

        # Targeted repair: the model sees its own output and exactly what is wrong
        # with it. This is why repair converges where a blind retry does not.
        messages = messages + [
            {"role": "assistant", "content": report.model_dump_json()},
            {"role": "user",
             "content": _REPAIR_TEMPLATE.format(
                 violations="\n".join(f"- {v}" for v in violations))},
        ]

    return ExtractionOutcome(doc_id, Status.QUARANTINED, None, violations, max_repairs, cost)


# ── Batch ─────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class BatchResult:
    outcomes: list[ExtractionOutcome]

    @property
    def usable(self) -> list[ExtractionOutcome]:
        return [o for o in self.outcomes if o.usable]

    @property
    def quarantined(self) -> list[ExtractionOutcome]:
        return [o for o in self.outcomes if o.status is Status.QUARANTINED]

    @property
    def mismatched(self) -> list[ExtractionOutcome]:
        return [o for o in self.outcomes if o.status is Status.MISMATCHED]

    @property
    def total_cost_usd(self) -> float:
        return sum(o.cost_usd for o in self.outcomes)

    def render(self) -> str:
        lines = [f"{'doc':<14}{'status':<14}{'passes':>7}{'cost':>10}  violations"]
        lines.append("-" * 78)
        for o in self.outcomes:
            first = o.violations[0] if o.violations else ""
            cost = f"${o.cost_usd:.5f}"
            lines.append(
                f"{o.doc_id:<14}{o.status.value:<14}{o.repair_passes:>7}"
                f"{cost:>10}  {first[:32]}"
            )
        lines.append("-" * 78)
        lines.append(
            f"{len(self.usable)} usable · {len(self.mismatched)} mismatched · "
            f"{len(self.quarantined)} quarantined · ${self.total_cost_usd:.5f}"
        )
        return "\n".join(lines)


def extract_batch(
    documents: Iterable[tuple[str, str]],
    *,
    client: ModelClient,
    truth_by_doc: dict[str, dict[str, Any]] | None = None,
    model: str = MODEL_SONNET,
    max_repairs: int = 2,
) -> BatchResult:
    """Extract many documents with per-record isolation.

    One malformed note must not abort the batch — a season's worth of game notes
    is exactly the kind of input where a handful are unusable and the rest are
    fine. Each record's failure is contained and reported.
    """
    truth_by_doc = truth_by_doc or {}
    outcomes: list[ExtractionOutcome] = []

    for doc_id, document in documents:
        try:
            outcomes.append(
                extract_game(doc_id, document, client=client,
                             truth=truth_by_doc.get(doc_id), model=model,
                             max_repairs=max_repairs)
            )
        except Exception as exc:  # noqa: BLE001 - per-record containment
            outcomes.append(
                ExtractionOutcome(doc_id, Status.QUARANTINED, None,
                                  [f"unhandled {type(exc).__name__}: {exc}"], 0, 0.0,
                                  error=str(exc))
            )

    return BatchResult(outcomes)
