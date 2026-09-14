"""Exercise 2 — a multi-agent tool with escalation logic.

The task: answer a coach's question about Highland or an opponent, using the
season data the dashboard already scrapes.

The interesting part is not the answering, it's *deciding the answer is good
enough*. A naive ladder escalates on low self-reported confidence alone, which
fails in both directions: a confidently wrong model never escalates, and an
overloaded endpoint escalates to a pricier model that is equally overloaded.

This ladder separates four independent questions:

  1. Should a model see this at all?  (`_policy_gate` — pre-flight, no API call)
  2. Did the call succeed?           (transient -> retry same tier; permanent -> stop)
  3. Is the answer well-formed?      (structured output validation)
  4. Is the answer *true*?           (`Grounder` — checks cited numbers against
                                      the real DataFrame, which is what actually
                                      catches a confident hallucination)

Only #3 and #4 — plus genuine low confidence — escalate to a more capable tier.
Everything else either retries in place, stops, or goes to a human.
"""

from __future__ import annotations

import dataclasses
import enum
import re
from typing import Any, Iterable

import pandas as pd
from pydantic import BaseModel, Field

from agent_lab.model import (
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
    ModelClient,
    ModelRequest,
    PermanentModelError,
    RefusalError,
    TransientModelError,
    call_with_retry,
)


# ── Structured answer contract ────────────────────────────────────────────────


class CitedStat(BaseModel):
    """One numeric claim the answer rests on, so it can be checked."""

    name: str = Field(description="Snake_case stat key, e.g. highland_ppg")
    value: float = Field(description="The numeric value being asserted")


class CoachAnswer(BaseModel):
    """What every tier must return, regardless of which model produced it."""

    answer: str = Field(description="Plain-language answer for a coach, 1-3 sentences")
    confidence: float = Field(ge=0.0, le=1.0, description="0-1 self-assessed confidence")
    stats_cited: list[CitedStat] = Field(
        default_factory=list,
        description="Every numeric claim made in the answer, so it can be verified",
    )
    needs_human: bool = Field(
        default=False, description="True if this should not be answered by a model"
    )


# ── Escalation reasons ────────────────────────────────────────────────────────


class Reason(str, enum.Enum):
    OK = "ok"
    SCHEMA_INVALID = "schema_invalid"
    LOW_CONFIDENCE = "low_confidence"
    UNGROUNDED = "ungrounded"
    SELF_FLAGGED = "self_flagged"
    REFUSED = "refused"
    BUDGET_EXCEEDED = "budget_exceeded"
    SENSITIVE_TOPIC = "sensitive_topic"
    TIER_EXHAUSTED = "tier_exhausted"
    PERMANENT_ERROR = "permanent_error"


#: Reasons that mean "a smarter model might do better". Anything not in this set
#: either resolves in place or goes straight to a human — escalating a refusal or
#: a malformed request just spends more money on the same outcome.
_ESCALATABLE = {
    Reason.SCHEMA_INVALID,
    Reason.LOW_CONFIDENCE,
    Reason.UNGROUNDED,
    Reason.SELF_FLAGGED,
}


# ── Grounding ─────────────────────────────────────────────────────────────────


class Grounder:
    """Checks a model's numeric claims against the real season data.

    This is the check that catches the failure mode self-reported confidence
    cannot: a model that states `highland_ppg = 61.2` with confidence 0.95 when
    the schedule says 45.5. Confidence is an opinion; this is arithmetic.
    """

    #: A cited value may differ from truth by this fraction and still pass —
    #: models legitimately round, and the underlying stats are themselves rounded.
    TOLERANCE = 0.02

    def __init__(self, facts: dict[str, float]) -> None:
        self.facts = facts

    @classmethod
    def from_schedule(cls, schedule: pd.DataFrame, prefix: str = "highland") -> "Grounder":
        """Derive checkable facts from a schedule frame (`scraper.get_schedule()`)."""
        played = schedule[schedule["result"].isin(["W", "L"])].dropna(subset=["hld_score"])
        if played.empty:
            return cls({})

        facts = {
            f"{prefix}_games_played": float(len(played)),
            f"{prefix}_wins": float((played["result"] == "W").sum()),
            f"{prefix}_losses": float((played["result"] == "L").sum()),
            f"{prefix}_ppg": round(float(played["hld_score"].mean()), 1),
            f"{prefix}_opp_ppg": round(float(played["opp_score"].mean()), 1),
            f"{prefix}_best_margin": float(played["margin"].max()),
            f"{prefix}_worst_margin": float(played["margin"].min()),
        }
        return cls(facts)

    def check(self, cited: Iterable[CitedStat]) -> list[str]:
        """Return a list of human-readable contradictions. Empty means grounded.

        Unknown stat names are *not* violations — the model may legitimately cite
        something this index does not track. Only a known key with a wrong value
        is a contradiction.
        """
        problems: list[str] = []
        for stat in cited:
            truth = self.facts.get(stat.name)
            if truth is None:
                continue
            tolerance = max(abs(truth) * self.TOLERANCE, 0.05)
            if abs(stat.value - truth) > tolerance:
                problems.append(
                    f"{stat.name}: claimed {stat.value}, actual {truth}"
                )
        return problems

    def as_context(self) -> str:
        """The fact sheet handed to the model, so grounding is achievable."""
        if not self.facts:
            return "No verified season facts are available."
        lines = "\n".join(f"  {k} = {v}" for k, v in sorted(self.facts.items()))
        return f"Verified season facts (cite these by exact key):\n{lines}"


# ── Policy gate ───────────────────────────────────────────────────────────────

#: Questions about a named player's health, or discipline/playing-time decisions,
#: are routed to a human before any model call. This is a youth-sports app: those
#: are decisions a coach owns, and in some cases protected information. A model
#: answering them confidently is worse than no answer.
_SENSITIVE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(injur\w*|concussion|hurt|medical|health|doctor|cleared to play)\b",
        r"\b(bench|benching|cut|kick(ed)? off|suspend\w*|discipline)\b",
        r"\b(should .{0,30}\bstart\b|playing time|minutes for)\b",
        r"\b(grades?|eligibility|academic\w*)\b",
    )
]


def _policy_gate(question: str) -> Reason | None:
    """Return a Reason if this question must not reach a model at all."""
    for pattern in _SENSITIVE_PATTERNS:
        if pattern.search(question):
            return Reason.SENSITIVE_TOPIC
    return None


# ── Ladder configuration ──────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Tier:
    model: str
    effort: str
    #: Below this confidence, escalate rather than accept.
    min_confidence: float


@dataclasses.dataclass(frozen=True)
class Policy:
    tiers: tuple[Tier, ...] = (
        # Cheap first pass. Held to a high confidence bar precisely because it is
        # cheap to escalate away from — a marginal Haiku answer is not worth
        # shipping to a coach when Sonnet costs fractions of a cent more.
        Tier(MODEL_HAIKU, effort="medium", min_confidence=0.85),
        Tier(MODEL_SONNET, effort="medium", min_confidence=0.70),
        # Last model tier: the bar drops because the alternative is a human.
        Tier(MODEL_OPUS, effort="high", min_confidence=0.55),
    )
    #: Hard ceiling for one question across every tier. Exceeding it stops the
    #: ladder and hands off to a human rather than silently spending more.
    budget_usd: float = 0.05
    max_retries_per_tier: int = 3


# ── Trace ─────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Attempt:
    model: str
    reason: Reason
    confidence: float | None
    cost_usd: float
    detail: str = ""


@dataclasses.dataclass
class Resolution:
    """The outcome, plus the full path that produced it."""

    answer: CoachAnswer | None
    reason: Reason
    attempts: list[Attempt] = dataclasses.field(default_factory=list)
    escalated_to_human: bool = False

    @property
    def total_cost_usd(self) -> float:
        return sum(a.cost_usd for a in self.attempts)

    @property
    def path(self) -> str:
        return " -> ".join(f"{a.model}:{a.reason.value}" for a in self.attempts) or "(no calls)"

    def summary(self) -> str:
        head = (
            f"[human] {self.reason.value}"
            if self.escalated_to_human
            else f"[answered by {self.attempts[-1].model}]"
            if self.attempts
            else "[no answer]"
        )
        return f"{head}  cost=${self.total_cost_usd:.5f}  path={self.path}"


# ── The ladder ────────────────────────────────────────────────────────────────

_SYSTEM = """You answer questions from a high school girls basketball coaching staff \
about the Highland Lady Bulldogs.

Rules:
- Answer only from the verified facts provided. Do not invent numbers.
- Every number in your answer must appear in stats_cited, using the exact fact key.
- Set confidence honestly. If the facts do not support an answer, say so and use a low
  confidence rather than guessing.
- Set needs_human=true for anything involving player health, discipline, playing time,
  or academic eligibility."""


def ask(
    question: str,
    *,
    client: ModelClient,
    grounder: Grounder,
    policy: Policy | None = None,
    sleep: Any = None,
) -> Resolution:
    """Answer a coach's question, escalating only when escalation can help."""
    policy = policy or Policy()
    attempts: list[Attempt] = []

    # 1. Pre-flight policy gate. No model sees a sensitive question.
    gate = _policy_gate(question)
    if gate is not None:
        return Resolution(answer=None, reason=gate, attempts=attempts, escalated_to_human=True)

    prompt = f"{grounder.as_context()}\n\nCoach's question: {question}"

    for tier in policy.tiers:
        # 2. Budget check happens *before* spending, not after.
        spent = sum(a.cost_usd for a in attempts)
        if spent >= policy.budget_usd:
            attempts.append(
                Attempt(tier.model, Reason.BUDGET_EXCEEDED, None, 0.0,
                        f"spent ${spent:.5f} of ${policy.budget_usd:.5f}")
            )
            return Resolution(None, Reason.BUDGET_EXCEEDED, attempts, escalated_to_human=True)

        request = ModelRequest(
            model=tier.model,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            effort=tier.effort,
            output_format=CoachAnswer,
            max_tokens=2048,
        )

        # 3. Execute. Transient failures retry in place; they are not evidence
        #    that this tier is incapable.
        retry_kwargs: dict[str, Any] = {"max_attempts": policy.max_retries_per_tier}
        if sleep is not None:
            retry_kwargs["sleep"] = sleep
        try:
            result = call_with_retry(client, request, **retry_kwargs)
        except TransientModelError as exc:
            attempts.append(Attempt(tier.model, Reason.TIER_EXHAUSTED, None, 0.0, str(exc)))
            continue  # Try the next tier — a different model may be less loaded.
        except RefusalError as exc:
            attempts.append(Attempt(tier.model, Reason.REFUSED, None, 0.0, str(exc)))
            return Resolution(None, Reason.REFUSED, attempts, escalated_to_human=True)
        except PermanentModelError as exc:
            attempts.append(Attempt(tier.model, Reason.PERMANENT_ERROR, None, 0.0, str(exc)))
            return Resolution(None, Reason.PERMANENT_ERROR, attempts, escalated_to_human=True)

        cost = result.cost_usd
        answer = result.parsed

        # 4. Well-formed?
        if not isinstance(answer, CoachAnswer):
            attempts.append(Attempt(tier.model, Reason.SCHEMA_INVALID, None, cost,
                                    "structured output missing or invalid"))
            continue

        # 5. Model itself says a human should handle it.
        if answer.needs_human:
            attempts.append(Attempt(tier.model, Reason.SELF_FLAGGED, answer.confidence, cost))
            return Resolution(answer, Reason.SELF_FLAGGED, attempts, escalated_to_human=True)

        # 6. True? This is the check that catches confident hallucination.
        problems = grounder.check(answer.stats_cited)
        if problems:
            attempts.append(Attempt(tier.model, Reason.UNGROUNDED, answer.confidence, cost,
                                    "; ".join(problems)))
            continue

        # 7. Confident enough for this tier?
        if answer.confidence < tier.min_confidence:
            attempts.append(Attempt(tier.model, Reason.LOW_CONFIDENCE, answer.confidence, cost,
                                    f"below {tier.min_confidence}"))
            continue

        attempts.append(Attempt(tier.model, Reason.OK, answer.confidence, cost))
        return Resolution(answer, Reason.OK, attempts)

    # Every tier tried and none produced a trustworthy answer.
    return Resolution(None, Reason.TIER_EXHAUSTED, attempts, escalated_to_human=True)
