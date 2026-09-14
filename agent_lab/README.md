# agent_lab — Claude Certified Architect Foundations practice

Four exercises, built against this repo's real data rather than a toy domain, so
the design decisions have actual constraints pushing back on them.

Everything here is **runnable offline**. No `ANTHROPIC_API_KEY` is set on this
machine and no `ant` CLI is installed, so every exercise runs against a
deterministic scripted client instead of the network. That is not a workaround —
it is the point. Escalation ladders and failure handling are *control flow*, and
control flow you cannot run is control flow you have not tested.

## Setup

```bash
.venv/Scripts/python.exe -m pip install -r requirements-agent-lab.txt
```

## Run them

```bash
.venv/Scripts/python.exe -m agent_lab.demo_escalation
.venv/Scripts/python.exe -m agent_lab.debug_research
.venv/Scripts/python.exe -m agent_lab.demo_extraction
```

## Layout

| File | Exercise | What it demonstrates |
|------|----------|----------------------|
| `model.py` | shared | The seam between agent logic and the network; per-tier request shaping; typed error taxonomy |
| `escalation.py` + `demo_escalation.py` | 2 | Multi-tier tool with escalation logic |
| `research.py` + `debug_research.py` | 3 | Fan-out/fan-in research pipeline, and the bugs found debugging it |
| `extraction.py` + `demo_extraction.py` | 4 | Schema → invariants → reconciliation → quarantine |

Exercise 1 (configuring Claude Code for a real project) lives outside this
package, in `CLAUDE.md` and `.claude/`.

## Exercise 2 — escalation

The ladder is Haiku → Sonnet → Opus → human, but the tier list is the least
interesting part. What matters is that four different questions are kept separate:

1. **Should a model see this at all?** A pre-flight policy gate routes questions
   about player health, discipline, playing time, or academic eligibility to a
   human with *zero* model calls. This is a youth-sports app; those are decisions
   a coach owns and in some cases protected information.
2. **Did the call succeed?** Transient failures (429, timeout, 5xx) retry *in
   place*. Escalating to a pricier model does not fix an overloaded endpoint.
3. **Is the answer well-formed?** Structured output validation.
4. **Is the answer true?** `Grounder` checks every cited number against the real
   schedule DataFrame.

Layer 4 is the one that earns its keep. Self-reported confidence cannot catch a
model that says `highland_ppg = 61.2` with confidence 0.95 when the data says
45.5 — confidence is an opinion, arithmetic is not. Only schema failure, low
confidence, and ungrounded claims escalate. Refusals and malformed requests stop
immediately, because a second model will produce the same outcome at higher cost.

To run the grounder against live data instead of fixtures:

```python
import scraper
from agent_lab.escalation import Grounder
g = Grounder.from_schedule(scraper.get_schedule())
```

## Exercise 3 — research pipeline, and its bugs

Three specialist workers fan out over disjoint slices of one opponent's season,
then an Opus synthesizer merges them. Workers see only their own slice: a
personnel worker that can see defensive splits will talk about them, badly, and
every extra token is paid for on every worker.

Two real defects surfaced when the adversarial harness was run, both of which the
design *claimed* not to have:

**1. One bad worker killed the entire brief.** `_run_worker` caught `ModelError`,
but `angle.slice_fn(bundle)` was called outside the `try`. A `ValueError` from one
slice function crossed the thread boundary and re-raised at `future.result()`,
taking down all three angles and the synthesis. Fixed with an explicit containment
boundary that catches `Exception` (but deliberately not `BaseException`, so Ctrl-C
still works).

**2. `degraded` was quietly lying.** A worker whose slice was empty returned a
valid report with zero findings, so `degraded` stayed `False` — a caller reading
that as "full coverage" would ship a brief where two of three angles rested on no
data at all. Split into `degraded` (an angle *failed*) and `thin_angles` (an angle
*reported nothing*), which are genuinely different conditions.

The harness also pins report ordering across 12 runs. Fan-out completion order is
nondeterministic; if reports were consumed in completion order the synthesizer
prompt would vary run to run and prompt caching would never hit.

## Exercise 4 — structured extraction

`extract → validate → repair → reconcile → accept | quarantine`.

The governing idea: **a schema is not a validator.** Pydantic guarantees an
integer sits where an integer belongs. It says nothing about whether a player can
make 4 field goals on 2 attempts. So validation widens in three layers:

1. **Schema** — types and ranges, free from Pydantic.
2. **Invariants** — basketball arithmetic. `points == 2*fg_made + three_made +
   ft_made` is an identity, so a violation is *proof* of a bad extraction with no
   judgment call involved. Threes are a subset of field goals; player points must
   sum to the team score; the result must agree with the scoreline.
3. **Reconciliation** — agreement with the scraped box score. This is the only
   layer that can catch a record that is internally flawless but describes the
   wrong game.

Layer 2 failures are **repairable**, and that distinction drives the whole design:
the violations name exactly what is wrong, so feeding them back gives the model
something specific to correct. A blind retry resends the same prompt and usually
reproduces the same error. Layer 3 failures are *not* repaired — the extraction
may be perfectly faithful to a document that is itself wrong — so they are
surfaced as `MISMATCHED` and marked unusable rather than "fixed" toward the box
score.

Anything still failing after the repair budget is quarantined **with its
reasons**. Never silently dropped, never silently accepted.
