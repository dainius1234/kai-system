# Parallel Dispatch Playbook

## When to dispatch in parallel
Only when workstreams pass both checks:

1. **Independence test (logical):** no task depends on outputs of the other.
2. **File-scope test (technical):** no overlapping files/modules likely to conflict.

If either test fails, run sequentially.

## Parallel dispatch steps
1. Identify candidate tasks and expected file scope for each.
2. Mark shared dependencies explicitly (if any).
3. Approve fan-out in `STATUS.md` notes.
4. Dispatch agents with explicit non-overlap guardrails.
5. Merge in dependency-safe order (if needed), then run post-merge checklist.

## Known approved pattern
- J6 identity files and MCP refactor are parallel-safe after J2 sequencing (Decision D13).
