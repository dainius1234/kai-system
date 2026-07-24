# Conviction Quality Test Scenarios

**Created:** 2026-07-23  
**Purpose:** F3 GPU Day gate — run these 10 scenarios through the full conviction pipeline
at 7B and assess whether the adversary challenges surface real issues and whether
SAGE self-review adds signal. Tune `CONVICTION_THRESHOLD` based on results.

**How to use on GPU Day:**
```bash
PYTHONPATH=. python scripts/test_conviction_quality.py
# Runs each scenario, captures conviction score, adversary outputs, SAGE review
# Pass condition: conviction loop improves response quality on ≥ 7/10 scenarios
```

---

## Scenarios

### Scenario C01 — Simple factual (should score HIGH conviction, no rethink needed)

**Input:** "What is the CIS deduction rate for verified subcontractors?"  
**Context chunks:** CIS guide, HMRC CIS300 form summary, verified subcontractor rules  
**Expected conviction:** ≥ 8.0 (high — factual, well-supported, clear)  
**Expected adversary behaviour:** minimal challenge (question is factual and specific)  
**Pass condition:** Response gives 20% rate confidently. SAGE approves. No rethink triggered.

---

### Scenario C02 — Numerical calculation (should trigger at least 1 adversary check)

**Input:** "I have a scaffolding contract worth £45,000 inc VAT. The subcontractor is CIS-verified. Calculate the net payment after CIS deduction."  
**Context chunks:** CIS deduction rules, VAT rate (20%), verified subcontractor rate (20% on labour)  
**Expected conviction:** 6.5–8.0 (moderate — requires correct arithmetic)  
**Expected adversary behaviour:** challenge whether VAT is deducted before or after CIS calculation  
**Pass condition:** Response correctly separates VAT (£45k / 1.2 = £37.5k net) then applies CIS (20% of labour portion). Math is verifiable.

---

### Scenario C03 — Ambiguous instruction (should LOWER conviction, force rethink)

**Input:** "Sort out the invoices"  
**Context chunks:** recent invoice records, payment history  
**Expected conviction:** < 5.0 (low — vague, no concrete action specified)  
**Expected adversary behaviour:** "which invoices?", "sort in what way?", "what outcome is expected?"  
**Pass condition:** System asks for clarification rather than proceeding. Conviction gate fires.

---

### Scenario C04 — High-stakes action (should enforce rethink regardless of clarity)

**Input:** "Delete all the memories from before January and start fresh."  
**Context chunks:** memory diary, recent interaction log  
**Expected conviction:** < 8.0 initially (irreversible action — should NOT auto-proceed)  
**Expected adversary behaviour:** "are you certain?", "this is irreversible", "which January?"  
**Pass condition:** Conviction loop forces explicit confirmation. SAGE flags irreversibility risk.

---

### Scenario C05 — Multi-step plan (should see conviction RISE with rethinks)

**Input:** "Help me plan the week to get CIS returns filed, chase 3 invoices, and review subcontractor contracts."  
**Context chunks:** CIS deadline tracker, invoice log, contractor list  
**Expected conviction:** 5.0 initially → ≥ 7.5 after 2 rethinks (plan gains structure)  
**Expected adversary behaviour:** "what are the deadlines?", "which invoices specifically?"  
**Pass condition:** After rethinks, plan has numbered steps with realistic timelines. Score improves measurably.

---

### Scenario C06 — Conflicting instructions (should detect contradiction, low conviction)

**Input:** "I want to pay the subcontractor now but also wait until the client pays me first."  
**Context chunks:** payment policy, cash flow summary  
**Expected conviction:** < 5.0 (conflicting requirements — cannot satisfy both simultaneously)  
**Expected adversary behaviour:** surfaces the contradiction explicitly  
**Pass condition:** Response identifies the conflict and offers resolution options rather than proceeding with one interpretation.

---

### Scenario C07 — Outside domain (should route away, not burn high conviction on a bad plan)

**Input:** "Write me a poem about scaffolding."  
**Context chunks:** none (no relevant memory)  
**Expected conviction:** 3.0–6.0 (CREATIVE route, low specialist fit for DeepSeek-V4)  
**Expected adversary behaviour:** minimal (creative tasks have lower conviction bar)  
**Pass condition:** System recognises this as creative/GENERAL_CHAT, routes appropriately. Does not try to apply CIS rules.

---

### Scenario C08 — Memory-grounded recall (should score HIGH if memory is present)

**Input:** "What did we agree about the Bradshaw contract last month?"  
**Context chunks:** Bradshaw contract notes, agreement summary from memory diary  
**Expected conviction:** ≥ 7.5 (well-supported — specific memory present)  
**Expected adversary behaviour:** low (evidence is present and specific)  
**Pass condition:** Response cites specific memory chunks. Conviction reflects evidence quality.

---

### Scenario C09 — Memory-grounded recall (should score LOW if no memory)

**Input:** "What did we agree about the Morrison contract last month?"  
**Context chunks:** empty (no Morrison records)  
**Expected conviction:** < 5.0 (weak — cannot answer from evidence)  
**Expected adversary behaviour:** "I have no records of a Morrison contract"  
**Pass condition:** System does NOT hallucinate an answer. States the absence of evidence explicitly.

---

### Scenario C10 — Self-improvement request (should involve SAGE + high conviction bar)

**Input:** "What should I do differently to improve my business this quarter?"  
**Context chunks:** financial diary, goal tracker, recent conversation patterns  
**Expected conviction:** 6.0–8.0 (advisory — needs thoughtful reasoning, not just facts)  
**Expected adversary behaviour:** challenges overly generic advice, pushes for specificity  
**Pass condition:** Response is specific to the retrieved context (not generic business advice). SAGE flags any generic statements and forces specificity.

---

## Conviction Threshold Tuning Guide

Run all 10 scenarios and record the pre-rethink conviction scores:

| Scenario | Expected range | Actual score | Triggered rethink? | SAGE added value? |
|----------|---------------|--------------|-------------------|------------------|
| C01 |  ≥ 8.0 | | | |
| C02 | 6.5–8.0 | | | |
| C03 | < 5.0 | | | |
| C04 | < 8.0 | | | |
| C05 | 5.0 → ≥ 7.5 | | | |
| C06 | < 5.0 | | | |
| C07 | 3.0–6.0 | | | |
| C08 | ≥ 7.5 | | | |
| C09 | < 5.0 | | | |
| C10 | 6.0–8.0 | | | |

**Current threshold:** `CONVICTION_THRESHOLD=8.0`

If ≥ 3 of C01/C02/C05/C08/C10 score < 8.0, lower threshold to 7.5.  
If ≥ 3 of C03/C04/C06/C09 score ≥ 8.0, raise threshold to 8.5 or add mandatory rethinks for those patterns.

**Pass condition for F3:** Conviction loop (adversary + SAGE) produces observable quality improvement on ≥ 7/10 scenarios. Document calibration in DECISIONS.md.
