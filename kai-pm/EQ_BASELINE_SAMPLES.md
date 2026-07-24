# EQ Baseline Samples — Emotional Intelligence Quality Gate

**Created:** 2026-07-23  
**Purpose:** F2 GPU Day gate — run these 20 samples through the EQ detection pipeline
at 7B and measure precision/recall vs expected labels. Set `EQ_CONFIDENCE_THRESHOLD`
based on results (target: ≥ 70% precision on known labels).

**How to use on GPU Day:**
```bash
PYTHONPATH=. python scripts/test_eq_quality.py
# Outputs per-sample classification + precision/recall summary
# Adjust EQ_CONFIDENCE_THRESHOLD in .env until precision >= 0.70
```

---

## Sample Set

### Positive Emotions

| ID | Input | Expected emotion | Expected intensity |
|----|-------|-----------------|-------------------|
| E01 | "Just got the contract signed. £85k job, starts Monday. Feeling good." | joy / relief | high |
| E02 | "The CIS return came back clean, no errors. That's a weight off." | relief | medium |
| E03 | "Finished the site visit early, weather held, lads did a great job. Proper proud of the team today." | pride / joy | high |
| E04 | "Got the mortgage approved. Years of work finally paying off." | joy / gratitude | high |
| E05 | "Client paid on time for once. Nice change." | satisfaction | low-medium |

### Negative Emotions — Stress / Frustration

| ID | Input | Expected emotion | Expected intensity |
|----|-------|-----------------|-------------------|
| E06 | "HMRC sent another letter. Don't even know what they want this time. Honestly sick of it." | frustration / anxiety | high |
| E07 | "Subbies keep ringing about payment but the main contractor hasn't paid me yet. What am I supposed to do." | frustration / stress | high |
| E08 | "Lost the tender again. Third time this month. Starting to wonder if it's worth bidding at all." | discouragement / self-doubt | high |
| E09 | "Material costs are through the roof and I can't pass it all on. Margin's going to be terrible." | stress / worry | medium-high |
| E10 | "Had to turn down work because I don't have enough lads. Good problems to have I suppose but stressful." | mixed: stress + positive framing | medium |

### Negative Emotions — Worry / Anxiety

| ID | Input | Expected emotion | Expected intensity |
|----|-------|-----------------|-------------------|
| E11 | "Not sure if I'm going to make payroll this month. Waiting on three invoices." | anxiety | high |
| E12 | "Haven't filed the VAT return yet. It was due last week. I know I need to sort it." | guilt / anxiety | medium |
| E13 | "The site inspector is coming Friday. There's a few things I'm not happy about." | worry / apprehension | medium |
| E14 | "Started getting chest pains last week. Think it's just stress but." | worry / concern | high |
| E15 | "Business is quiet. Really quiet. Nothing in the diary after next month." | anxiety / uncertainty | high |

### Complex / Mixed

| ID | Input | Expected emotion | Expected intensity |
|----|-------|-----------------|-------------------|
| E16 | "Turned down the big contract. Good money but wrong kind of work. No regrets but still." | ambivalence / mild regret | medium |
| E17 | "The lad I trained up just started his own company. Should be proud but it stings a bit." | bittersweet / mixed pride | medium |
| E18 | "Mam's not well. Hard to concentrate on work. Just going through the motions." | grief / distraction | high |
| E19 | "Thinking about taking on a business partner. Excited but also nervous as hell." | anticipation + anxiety | medium-high |
| E20 | "Quiet day. Just did some paperwork. Nothing to report really." | neutral / low affect | low |

---

## Evaluation Rubric

For each sample, the EQ detector should return:
- `emotion` (primary label)
- `intensity` (low / medium / high)
- `confidence` (0.0–1.0)

**Scoring:**
- Correct emotion category (±1 adjacent label allowed): +1
- Correct intensity (±1 level allowed): +1

| Threshold | Precision target | Action |
|-----------|-----------------|--------|
| EQ_CONFIDENCE_THRESHOLD = 0.6 | Measure precision/recall | Start here |
| EQ_CONFIDENCE_THRESHOLD = 0.7 | Likely better precision | Tighten if recall holds |
| EQ_CONFIDENCE_THRESHOLD = 0.8 | High precision / lower recall | Use for critical paths only |

**Pass condition:** ≥ 14/20 correct emotion category (70%) at chosen threshold.  
Document results in DECISIONS.md (Phase 1 entry: F2 complete).

---

## Notes

- Samples are from the CIS/construction context — domain-specific vocabulary matters
- E10, E16, E17 are intentionally ambiguous — a 7B model handling these correctly is a good quality signal
- E20 (neutral) must NOT be labelled as emotional — false positives on neutral are costly
- Run at both 0.5b and 7b to measure the quality delta explicitly
