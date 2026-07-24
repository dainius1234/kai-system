# 0.5b Baseline Responses

**Captured:** *(pending — run `make capture-baseline` against your local dev stack)*  
**Model:** qwen2.5:0.5b  
**Purpose:** G4 comparison baseline — compare these against 7B responses on GPU Day  

---

## How to populate this file

Run the following against your local dev stack **before the GPU arrives**:

```bash
# Ensure the dev stack is up and using qwen2.5:0.5b (the default)
make core-up   # or docker compose -f docker-compose.minimal.yml up -d

# Wait for agentic to be healthy
curl -s http://localhost:8007/health

# Capture the baseline
make capture-baseline
# Output is written here automatically (overwrites this stub)
```

The script sends these 5 prompts and captures the full streaming responses:

| ID | Label |
|----|-------|
| P1 | Factual recall — CIS deduction rate for verified subcontractors |
| P2 | Multi-step numerical reasoning — £45k scaffolding contract net payment |
| P3 | Memory and context — VAT threshold + CIS interaction |
| P4 | Self-awareness — honest capability statement |
| P5 | Complex task decomposition — week planning for CIS/invoices/contracts |

---

## G4 Evaluation Guide

On GPU Day, run the same 5 prompts against the 7B stack and fill in the scoring table:

| Prompt | Completeness | Numerical accuracy | Structure |
|--------|-------------|-------------------|-----------|
| P1 |  |  |  |
| P2 |  |  |  |
| P3 |  |  |  |
| P4 |  |  |  |
| P5 |  |  |  |

**Pass condition:** ≥3/5 prompts score better on at least 2 of 3 axes.  
Document the delta in DECISIONS.md D83 (Phase 1 entry declaration).
