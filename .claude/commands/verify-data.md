---
description: Run a full data-integrity sweep of the scraper and parity flags
allowed-tools: Read, Grep, Bash, Task
---

Run a data-integrity sweep before trusting the dashboard.

Launch both project subagents concurrently — they are independent:

- `scrape-verifier` — confirm the schedule and box-score parsing still match the live site
- `parity-auditor` — confirm `OPPONENT_METADATA` flags match the columns actually scraped

Then combine their findings into one short report:

- **Green** — counts match and flags are verified
- **Yellow** — data loads but something is unverified (e.g. network sampling was partial)
- **Red** — games are being silently skipped, or a parity flag contradicts observed data

Lead with the verdict. A coach reads this before a game; the headline matters more than
the detail. Do not fix anything in this command — report only, and list the proposed
edits for a separate, deliberate change.
