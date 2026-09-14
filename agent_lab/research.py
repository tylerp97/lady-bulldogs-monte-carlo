"""Exercise 3 — a multi-agent research pipeline (design half).

Goal: turn one opponent's scraped season into a coach-facing scouting brief.

    plan  ->  fan out to specialist workers  ->  synthesize

                      ┌── offense  (Sonnet) ──┐
    planner (Opus) ───┼── defense  (Sonnet) ──┼──> synthesizer (Opus) -> brief
                      └── personnel (Haiku) ──┘

Four design decisions carry most of the weight:

1. **Context isolation.** Each worker receives only its own slice of facts, not
   the whole bundle. A personnel worker that can see defensive splits will talk
   about them, badly, and every extra token is paid for on every worker.

2. **Partial failure is normal.** Opponent data is scraped from a site that
   silently omits stats (see `OPPONENT_METADATA` in `scraper.py`). A worker whose
   slice is empty must degrade to "no data" rather than invent findings, and one
   dead worker must not lose the brief.

3. **Contradictions are the product, not an error.** Two workers looking at
   different slices routinely disagree ("they collapse under pressure" vs "they
   protect the ball"). The synthesizer surfaces the disagreement instead of
   silently picking one.

4. **Everything is traced.** `Trace` records a span per model call — model, cost,
   outcome, error. Without it a fan-out failure is invisible: you get a shorter
   brief and no indication that a third of the research never happened.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import threading
import time
from typing import Any, Callable

from pydantic import BaseModel, Field

from agent_lab.model import (
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
    ModelClient,
    ModelError,
    ModelRequest,
    ModelResult,
)


# ── Contracts ─────────────────────────────────────────────────────────────────


class Finding(BaseModel):
    claim: str = Field(description="One scouting claim, stated for a coach")
    evidence_keys: list[str] = Field(
        default_factory=list, description="Fact keys from the provided slice that support it"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    severity: str = Field(description="high | medium | low — gameplan priority")


class WorkerReport(BaseModel):
    angle: str
    findings: list[Finding] = Field(default_factory=list)
    data_gaps: list[str] = Field(
        default_factory=list, description="What this angle could not assess and why"
    )


class ScoutingBrief(BaseModel):
    headline: str = Field(description="One sentence a coach reads first")
    priorities: list[str] = Field(description="Ranked gameplan priorities")
    contradictions: list[str] = Field(
        default_factory=list, description="Disagreements between angles, left unresolved"
    )
    coverage_note: str = Field(
        default="", description="What the brief could not cover, and why"
    )


# ── Tracing ───────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Span:
    name: str
    model: str
    ok: bool
    cost_usd: float
    seconds: float
    error: str = ""


class Trace:
    """Thread-safe span recorder. The fan-out writes to this concurrently."""

    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._lock = threading.Lock()

    def record(self, span: Span) -> None:
        with self._lock:
            self.spans.append(span)

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    @property
    def failures(self) -> list[Span]:
        return [s for s in self.spans if not s.ok]

    def render(self) -> str:
        lines = [f"{'span':<22}{'model':<20}{'ok':<5}{'cost':>9}  {'sec':>5}"]
        lines.append("-" * 68)
        for s in self.spans:
            lines.append(
                f"{s.name:<22}{s.model:<20}{'yes' if s.ok else 'NO':<5}"
                f"${s.cost_usd:>8.5f}  {s.seconds:>5.2f}"
                + (f"  {s.error}" if s.error else "")
            )
        lines.append("-" * 68)
        lines.append(f"{'TOTAL':<42}${self.total_cost_usd:>8.5f}")
        return "\n".join(lines)


# ── Research angles ───────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Angle:
    """One specialist worker: a question, a model, and a slice of the data."""

    key: str
    model: str
    effort: str
    question: str
    #: Pulls this angle's facts out of the full bundle. Context isolation lives here.
    slice_fn: Callable[[dict[str, Any]], dict[str, Any]]


def _slice_offense(bundle: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in bundle.items() if k.startswith(("scoring_", "shooting_"))}


def _slice_defense(bundle: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in bundle.items() if k.startswith(("allowed_", "turnover_"))}


def _slice_personnel(bundle: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in bundle.items() if k.startswith("player_")}


DEFAULT_ANGLES: tuple[Angle, ...] = (
    Angle("offense", MODEL_SONNET, "medium",
          "How does this team score, and what suppresses it?", _slice_offense),
    Angle("defense", MODEL_SONNET, "medium",
          "How does this team defend, and where does it break down?", _slice_defense),
    # Personnel is mostly lookup over a small table — the cheap tier is enough.
    Angle("personnel", MODEL_HAIKU, "medium",
          "Who must be accounted for, and what happens when they are contained?",
          _slice_personnel),
)


# ── Result ────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class ResearchResult:
    opponent: str
    brief: ScoutingBrief | None
    reports: list[WorkerReport]
    trace: Trace
    #: An angle failed outright and filed nothing.
    degraded: bool = False

    @property
    def thin_angles(self) -> list[str]:
        """Angles that reported successfully but produced zero findings.

        Distinct from `degraded`. A worker whose slice was empty returns a valid
        report with data gaps and no findings — nothing failed, but coverage is
        thinner than the angle count suggests. Collapsing the two would let a
        caller read `degraded == False` as "full coverage" when two thirds of the
        brief rests on no data.
        """
        return [r.angle for r in self.reports if not r.findings]

    def summary(self) -> str:
        state = "DEGRADED" if self.degraded else "complete"
        thin = self.thin_angles
        thin_note = f", {len(thin)} thin ({', '.join(thin)})" if thin else ""
        return (
            f"{self.opponent}: {state}, {len(self.reports)} angle(s) reported{thin_note}, "
            f"${self.trace.total_cost_usd:.5f}"
        )


# ── Pipeline ──────────────────────────────────────────────────────────────────

_WORKER_SYSTEM = """You are a {angle} specialist scouting a high school girls basketball \
opponent for the Highland coaching staff.

You can see only the {angle} slice of the data. That is deliberate — do not speculate about \
anything outside it.

Rules:
- Ground every finding in the fact keys provided. List them in evidence_keys.
- If the slice is empty or too thin to support a finding, return no findings and explain \
what is missing in data_gaps. An honest gap is more useful to a coach than a guess.
- severity is how much it should shape the gameplan: high, medium, or low."""

_SYNTH_SYSTEM = """You are the head scout. Several specialists each saw one slice of an \
opponent's season and filed findings.

Write the brief the coaching staff reads before the game.

Rules:
- Rank priorities by what actually changes the gameplan.
- Where two angles disagree, put the disagreement in contradictions and do NOT silently \
pick a winner. A coach needs to know the read is contested.
- If an angle is missing or reported only data gaps, say so in coverage_note. Never imply \
coverage you do not have."""


def _run_worker(
    angle: Angle,
    opponent: str,
    bundle: dict[str, Any],
    client: ModelClient,
    trace: Trace,
) -> WorkerReport | None:
    """Run one angle inside a containment boundary.

    Returns None on any failure. The broad `except Exception` is deliberate: this
    function runs on a pool thread, and an exception escaping it re-raises at
    `future.result()` on the main thread, taking down the entire brief. A bug in
    one angle's `slice_fn` must cost that angle, not the whole scouting report.
    `BaseException` is intentionally *not* caught — Ctrl-C should still work.
    """
    try:
        return _run_worker_inner(angle, opponent, bundle, client, trace)
    except Exception as exc:  # noqa: BLE001 - containment boundary, see docstring
        trace.record(Span(f"worker:{angle.key}", angle.model, False, 0.0, 0.0,
                          f"unhandled {type(exc).__name__}: {exc}"))
        return None


def _run_worker_inner(
    angle: Angle,
    opponent: str,
    bundle: dict[str, Any],
    client: ModelClient,
    trace: Trace,
) -> WorkerReport | None:
    facts = angle.slice_fn(bundle)

    if not facts:
        # No model call at all: an empty slice has a known answer, and paying a
        # model to say "I have no data" is pure waste.
        trace.record(Span(f"worker:{angle.key}", "(skipped)", True, 0.0, 0.0,
                          "empty slice — no model call"))
        return WorkerReport(
            angle=angle.key,
            findings=[],
            data_gaps=[f"No {angle.key} data was scraped for {opponent}."],
        )

    fact_lines = "\n".join(f"  {k} = {v}" for k, v in sorted(facts.items()))
    request = ModelRequest(
        model=angle.model,
        system=_WORKER_SYSTEM.format(angle=angle.key),
        messages=[{
            "role": "user",
            "content": (
                f"Opponent: {opponent}\n"
                f"Question: {angle.question}\n\n"
                f"{angle.key} facts:\n{fact_lines}"
            ),
        }],
        effort=angle.effort,
        output_format=WorkerReport,
        max_tokens=2048,
    )

    started = time.monotonic()
    try:
        result: ModelResult = client.complete(request)
    except ModelError as exc:
        trace.record(Span(f"worker:{angle.key}", angle.model, False, 0.0,
                          time.monotonic() - started, str(exc)))
        return None

    elapsed = time.monotonic() - started
    report = result.parsed
    if not isinstance(report, WorkerReport):
        trace.record(Span(f"worker:{angle.key}", angle.model, False, result.cost_usd,
                          elapsed, "structured output invalid"))
        return None

    # The model chooses the angle label; force it to the real one so downstream
    # grouping cannot be corrupted by a mislabeled report.
    report.angle = angle.key
    trace.record(Span(f"worker:{angle.key}", angle.model, True, result.cost_usd, elapsed))
    return report


def research_opponent(
    opponent: str,
    bundle: dict[str, Any],
    *,
    client: ModelClient,
    angles: tuple[Angle, ...] = DEFAULT_ANGLES,
    max_workers: int = 3,
    min_reports: int = 1,
) -> ResearchResult:
    """Fan out across research angles, then synthesize one brief."""
    trace = Trace()

    # ── Fan out ───────────────────────────────────────────────────────────────
    reports: list[WorkerReport] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_worker, angle, opponent, bundle, client, trace): angle
            for angle in angles
        }
        for future in concurrent.futures.as_completed(futures):
            report = future.result()
            if report is not None:
                reports.append(report)

    # Deterministic order regardless of completion order — otherwise the
    # synthesizer prompt varies run to run and prompt caching never hits.
    order = {angle.key: i for i, angle in enumerate(angles)}
    reports.sort(key=lambda r: order.get(r.angle, 99))

    degraded = len(reports) < len(angles)

    if len(reports) < min_reports:
        trace.record(Span("synthesize", "(skipped)", False, 0.0, 0.0,
                          f"only {len(reports)} report(s), need {min_reports}"))
        return ResearchResult(opponent, None, reports, trace, degraded=True)

    # ── Synthesize ────────────────────────────────────────────────────────────
    missing = [a.key for a in angles if a.key not in {r.angle for r in reports}]
    payload = "\n\n".join(
        f"### {r.angle}\n" + (
            "\n".join(
                f"- [{f.severity}] {f.claim} (confidence {f.confidence}, "
                f"evidence: {', '.join(f.evidence_keys) or 'none'})"
                for f in r.findings
            ) or "- (no findings)"
        ) + (
            "\ndata gaps: " + "; ".join(r.data_gaps) if r.data_gaps else ""
        )
        for r in reports
    )
    if missing:
        payload += f"\n\n### MISSING ANGLES\nThese angles failed and filed nothing: {', '.join(missing)}"

    request = ModelRequest(
        model=MODEL_OPUS,
        system=_SYNTH_SYSTEM,
        messages=[{"role": "user", "content": f"Opponent: {opponent}\n\n{payload}"}],
        effort="high",
        output_format=ScoutingBrief,
        max_tokens=4096,
    )

    started = time.monotonic()
    try:
        result = client.complete(request)
    except ModelError as exc:
        trace.record(Span("synthesize", MODEL_OPUS, False, 0.0,
                          time.monotonic() - started, str(exc)))
        return ResearchResult(opponent, None, reports, trace, degraded=True)

    elapsed = time.monotonic() - started
    brief = result.parsed
    if not isinstance(brief, ScoutingBrief):
        trace.record(Span("synthesize", MODEL_OPUS, False, result.cost_usd, elapsed,
                          "structured output invalid"))
        return ResearchResult(opponent, None, reports, trace, degraded=True)

    trace.record(Span("synthesize", MODEL_OPUS, True, result.cost_usd, elapsed))
    return ResearchResult(opponent, brief, reports, trace, degraded=degraded)
