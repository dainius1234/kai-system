# Kai PM Decision Log

**Rule:** This file is append-only. Never edit past entries; supersede with a new numbered entry.

## D1 — 2026-04-20
- **Context:** Delivery work was happening without a single durable PM source.
- **Decision:** Run Kai in a locked-sequence delivery model rather than ad-hoc task hopping.
- **Rationale:** Prevents context drift and makes priorities auditable.
- **Consequences:** Every meaningful PR must map to sequence status.

## D2 — 2026-04-20
- **Context:** Runtime and docs had diverged in previous cycles.
- **Decision:** Keep docs synchronization as a merge-gate expectation, not optional cleanup.
- **Rationale:** Documentation is operational state, not a side artifact.
- **Consequences:** Post-merge updates to STATUS/README/CHANGELOG become mandatory.

## D3 — 2026-04-20
- **Context:** Kai is built jewel-by-jewel with explicit ordering needs.
- **Decision:** Keep J1–J7 as named delivery units inside the sequence.
- **Rationale:** Shared naming shortens planning and review loops.
- **Consequences:** PRs must identify jewel scope or explicitly declare N/A.

## D4 — 2026-04-20
- **Context:** Hardware availability is the hard external constraint.
- **Decision:** Mark Phases 1, 2, 4, and 5 as blocked on GPU procurement.
- **Rationale:** Removes false urgency and keeps software-only work focused.
- **Consequences:** Queue independent software tasks while hardware is pending.

## D5 — 2026-04-20
- **Context:** Recovery between sessions was costly and error-prone.
- **Decision:** Maintain a dedicated bootstrap document for rapid re-hydration.
- **Rationale:** Cuts session startup time and preserves operator continuity.
- **Consequences:** Resume flow always starts from a canonical bootstrap file.

## D6 — 2026-04-20
- **Context:** Risk handling was implicit.
- **Decision:** Maintain an explicit PM risk register with owners and mitigations.
- **Rationale:** Visible risks are easier to manage than implicit assumptions.
- **Consequences:** New material risks must be logged with status.

## D7 — 2026-04-20
- **Context:** Technology choices were discussed but not tracked over time.
- **Decision:** Use a rolling tech radar with Adopt/Trial/Assess/Hold states.
- **Rationale:** Structured evaluation avoids random tool churn.
- **Consequences:** Tech proposals route through a standard evaluation playbook.

## D8 — 2026-04-20
- **Context:** PM quality depended too heavily on manual reminders.
- **Decision:** Add lightweight GitHub-native PM guardrails (templates/workflows).
- **Rationale:** Automation catches drift between human check-ins.
- **Consequences:** PRs/issues/workflows become part of PM operating system.

## D9 — 2026-04-21
- **Context:** PM state was spread across ad-hoc chat and root docs.
- **Decision:** Adopt `kai-pm/` as the durable PM brain directory.
- **Rationale:** Centralizes state, playbooks, risks, decisions, and resume context.
- **Consequences:** `PROJECT_STATUS.md` is reduced to a redirect pointer.

## D10 — 2026-04-21
- **Context:** Independent workstreams were dispatched serially.
- **Decision:** Adopt parallel agent dispatch for independent scopes.
- **Rationale:** Parallelism reduces wall-clock delivery time without increasing coupling.
- **Consequences:** Independence and file-scope checks are required before fan-out.

## D11 — 2026-04-21
- **Context:** PM process checks were inconsistent across pull requests.
- **Decision:** Add `.github` PM automation (PR template, status workflow, drift detector, tech-watch reminder, issue templates).
- **Rationale:** Converts PM policy into repeatable repo automation.
- **Consequences:** PM hygiene checks run continuously, not only during manual review.

## D12 — 2026-04-21
- **Context:** Proposed sub-agents (Doc-Bot/Test-Bot/Bench-Bot/Health-Bot) add coordination overhead.
- **Decision:** Defer sub-agent rollout until explicit unlock triggers are met.
- **Rationale:** Avoid tool/process sprawl before the baseline PM system stabilizes.
- **Consequences:** Revisit when throughput bottlenecks justify additional bots.

## D13 — 2026-04-21
- **Context:** J6 identity work and MCP refactor are independent by file scope.
- **Decision:** After J2 merges, dispatch J6 and MCP refactor in parallel.
- **Rationale:** Safe concurrency with no logical dependency accelerates roadmap.
- **Consequences:** Parallel dispatch becomes the default for proven-independent tasks.
