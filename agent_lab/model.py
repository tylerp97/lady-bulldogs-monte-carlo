"""Model access layer shared by the three agent exercises.

Every model call in `agent_lab` goes through the `ModelClient` protocol. That
lets the identical pipeline code run against the real Anthropic API or against a
deterministic `ScriptedClient` in tests — which is the only practical way to
exercise escalation ladders and failure branches without spending money on every
run, or needing network access at all.

Model-specific request shaping lives here and nowhere else. The rules are not
uniform across the tiers, and getting them wrong is a 400 rather than a
degradation:

  * Haiku 4.5 rejects `output_config.effort`, and uses the older
    `thinking={"type": "enabled", "budget_tokens": N}` form rather than adaptive
    thinking. We simply omit both — the cheap tier is meant to be cheap.
  * Sonnet 5 and Opus 5 take adaptive thinking plus an `effort` level.
  * Opus 5 runs adaptive thinking by default; passing it explicitly is equivalent
    and keeps the call self-documenting.
"""

from __future__ import annotations

import dataclasses
import random
import time
from typing import Any, Protocol, Sequence

# ── Model tiers ───────────────────────────────────────────────────────────────

MODEL_HAIKU = "claude-haiku-4-5"
MODEL_SONNET = "claude-sonnet-5"
MODEL_OPUS = "claude-opus-5"

#: USD per 1M tokens, (input, output). Used for budget enforcement, not billing.
PRICING: dict[str, tuple[float, float]] = {
    MODEL_HAIKU: (1.00, 5.00),
    MODEL_SONNET: (2.00, 10.00),
    MODEL_OPUS: (5.00, 25.00),
}

#: Models that accept `output_config.effort` and adaptive thinking.
_SUPPORTS_EFFORT = {MODEL_SONNET, MODEL_OPUS}


# ── Errors ────────────────────────────────────────────────────────────────────


class ModelError(RuntimeError):
    """Base class for model-call failures."""


class TransientModelError(ModelError):
    """Retryable: rate limit, timeout, 5xx, connection reset.

    The caller should retry the *same* tier with backoff. Escalating to a more
    expensive model does not fix an overloaded endpoint.
    """


class PermanentModelError(ModelError):
    """Not retryable: malformed request, bad model id, auth failure.

    Retrying or escalating burns money without changing the outcome. Fail fast.
    """


class RefusalError(ModelError):
    """The model declined the request (`stop_reason == "refusal"`).

    Distinct from both error classes above: the call succeeded and was billed,
    but there is no answer. Escalation policy decides whether a different tier
    should see it — for most content categories, it should not.
    """

    def __init__(self, message: str, category: str | None = None) -> None:
        super().__init__(message)
        self.category = category


# ── Request / result ──────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class ModelRequest:
    """One model call, independent of which client executes it."""

    model: str
    system: str
    messages: list[dict[str, Any]]
    max_tokens: int = 4096
    effort: str = "medium"
    #: A Pydantic model class. When set, the response is parsed and validated
    #: against it via structured outputs.
    output_format: type | None = None
    tools: Sequence[dict[str, Any]] | None = None


@dataclasses.dataclass
class ModelResult:
    """What came back, plus what it cost."""

    model: str
    text: str
    parsed: Any | None
    stop_reason: str | None
    input_tokens: int
    output_tokens: int

    @property
    def cost_usd(self) -> float:
        in_rate, out_rate = PRICING.get(self.model, (0.0, 0.0))
        return (self.input_tokens * in_rate + self.output_tokens * out_rate) / 1_000_000


class ModelClient(Protocol):
    """The seam between agent logic and the network."""

    def complete(self, request: ModelRequest) -> ModelResult: ...


# ── Real client ───────────────────────────────────────────────────────────────


class AnthropicClient:
    """Calls the real Messages API via the official SDK."""

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            import anthropic

            # Zero-arg construction resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
            # or an `ant auth login` profile — do not hardcode a key here.
            client = anthropic.Anthropic()
        self._client = client

    def _kwargs(self, request: ModelRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": request.messages,
        }
        if request.model in _SUPPORTS_EFFORT:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": request.effort}
        if request.tools:
            kwargs["tools"] = list(request.tools)
        return kwargs

    def complete(self, request: ModelRequest) -> ModelResult:
        import anthropic

        kwargs = self._kwargs(request)
        try:
            if request.output_format is not None:
                # `parse` builds the JSON schema from the Pydantic type, sends it as
                # a structured output, and validates the response before returning.
                response = self._client.messages.parse(
                    output_format=request.output_format, **kwargs
                )
                parsed = response.parsed_output
            else:
                response = self._client.messages.create(**kwargs)
                parsed = None
        # Most specific first: a single `except APIStatusError` would collapse the
        # retryable / non-retryable distinction that the escalation ladder depends on.
        except anthropic.NotFoundError as exc:
            raise PermanentModelError(f"unknown model {request.model!r}: {exc}") from exc
        except anthropic.BadRequestError as exc:
            raise PermanentModelError(f"malformed request: {exc}") from exc
        except anthropic.AuthenticationError as exc:
            raise PermanentModelError(f"auth failed: {exc}") from exc
        except anthropic.PermissionDeniedError as exc:
            raise PermanentModelError(f"permission denied: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise TransientModelError(f"rate limited: {exc}") from exc
        except anthropic.APITimeoutError as exc:
            raise TransientModelError(f"timed out: {exc}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise TransientModelError(f"server error {exc.status_code}") from exc
            raise PermanentModelError(f"api error {exc.status_code}: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise TransientModelError(f"connection error: {exc}") from exc

        # A refusal is HTTP 200 — it never raises. Check before reading content.
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            raise RefusalError(f"model declined ({category})", category=category)

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return ModelResult(
            model=response.model,
            text=text,
            parsed=parsed,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


# ── Scripted client (offline) ─────────────────────────────────────────────────


@dataclasses.dataclass
class ScriptedTurn:
    """One canned outcome. Either a result payload or an exception to raise."""

    text: str = ""
    parsed: Any | None = None
    raises: Exception | None = None
    input_tokens: int = 1200
    output_tokens: int = 300
    stop_reason: str = "end_turn"


class ScriptedClient:
    """Deterministic offline stand-in for `AnthropicClient`.

    Responses are keyed by model id so a test can say "Haiku returns garbage,
    Sonnet returns a low-confidence answer, Opus gets it right" and then assert
    on the escalation path that produced the final result.

    Every call is appended to `calls`, which is the trace the exercises assert on.
    """

    def __init__(self, script: dict[str, list[ScriptedTurn]] | None = None) -> None:
        self.script = script or {}
        self.calls: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResult:
        self.calls.append(request)

        queue = self.script.get(request.model)
        if not queue:
            raise PermanentModelError(
                f"ScriptedClient has no turn queued for {request.model!r} "
                f"(call #{len(self.calls)})"
            )

        turn = queue.pop(0)
        if turn.raises is not None:
            raise turn.raises

        return ModelResult(
            model=request.model,
            text=turn.text,
            parsed=turn.parsed,
            stop_reason=turn.stop_reason,
            input_tokens=turn.input_tokens,
            output_tokens=turn.output_tokens,
        )


# ── Retry helper ──────────────────────────────────────────────────────────────


def call_with_retry(
    client: ModelClient,
    request: ModelRequest,
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    sleep: Any = time.sleep,
) -> ModelResult:
    """Retry *transient* failures at the same tier, with jittered backoff.

    The SDK already retries 429/5xx twice by default; this wraps that for the
    cases the SDK gives up on, and — more importantly — keeps transient failures
    from being misread as "this tier can't do the job", which would escalate to a
    more expensive model for no reason.
    """
    last: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return client.complete(request)
        except TransientModelError as exc:
            last = exc
            if attempt == max_attempts - 1:
                break
            sleep(base_delay * (2**attempt) + random.uniform(0, 0.25))
    raise last  # type: ignore[misc]


class RoutingScriptedClient:
    """Scripted client that routes by an arbitrary tag, not by model id.

    `ScriptedClient` keys its queues by model, which breaks the moment two
    concurrent workers share a tier: whichever thread arrives first pops the
    other's scripted turn, and the test fails for a reason that has nothing to do
    with the code under test. This routes on a caller-supplied function of the
    request instead, and holds a lock so concurrent pops stay deterministic.
    """

    def __init__(
        self,
        script: dict[str, list[ScriptedTurn]],
        router: Any,
    ) -> None:
        self.script = {k: list(v) for k, v in script.items()}
        self.router = router
        self.calls: list[tuple[str, ModelRequest]] = []
        self._lock = __import__("threading").Lock()

    def complete(self, request: ModelRequest) -> ModelResult:
        tag = self.router(request)
        with self._lock:
            self.calls.append((tag, request))
            queue = self.script.get(tag)
            if not queue:
                raise PermanentModelError(f"no scripted turn for tag {tag!r}")
            turn = queue.pop(0)

        if turn.raises is not None:
            raise turn.raises

        return ModelResult(
            model=request.model,
            text=turn.text,
            parsed=turn.parsed,
            stop_reason=turn.stop_reason,
            input_tokens=turn.input_tokens,
            output_tokens=turn.output_tokens,
        )

    def tags_called(self) -> list[str]:
        return [tag for tag, _ in self.calls]
