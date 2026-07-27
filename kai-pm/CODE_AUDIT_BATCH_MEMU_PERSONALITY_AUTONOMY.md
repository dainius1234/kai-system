# Kai Code Audit — memU Personality, Conscience and Proactive Autonomy Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch covers the P17–P22 emotional intelligence, narrative identity, imagination, conscience, scheduling and operator-model subsystems in `memu-core/app.py`, with external-delivery confirmation against `supervisor/app.py`. The general unauthenticated memU Core control plane is already recorded in `CODE_AUDIT_BATCH_MEMU_CORE_HOT_PATH.md` and is not counted again here.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-PERSONA-001 | CRITICAL | Anonymous feedback can create high-authority positive or correction memories from any session |
| KAI-PERSONA-002 | CRITICAL | Anonymous value/conscience calls can rewrite the moral model and fabricate integrity injected into Agentic prompts |
| KAI-PERSONA-003 | HIGH | Anonymous reminders are delivered to the operator through Telegram |
| KAI-PERSONA-004 | HIGH | Anonymous scheduled-task titles are delivered to the operator through Telegram |
| KAI-PERSONA-005 | HIGH | Anonymous nudge escalation can trigger tough-love and intervention Telegram messages |
| KAI-PERSONA-006 | HIGH | Feedback has no authenticated session ownership or response identity |
| KAI-PERSONA-007 | HIGH | Negative feedback indices select arbitrary recent messages and invalid negatives fail silently |
| KAI-PERSONA-008 | HIGH | Raw feedback comments are copied into correction memories without the sanitised value |
| KAI-PERSONA-009 | HIGH | Feedback state is global, volatile and worker-local |
| KAI-PERSONA-010 | HIGH | Feedback returns success even when the memory side effect fails |
| KAI-PERSONA-011 | HIGH | Feedback statistics disclose recent comments, ratings and session identifiers |
| KAI-PERSONA-012 | HIGH | Emotion detection is manipulable substring matching without negation or attribution handling |
| KAI-PERSONA-013 | HIGH | Emotional state from every caller/session is merged into one operator timeline |
| KAI-PERSONA-014 | HIGH | Emotional timeline and echo APIs disclose trigger-message snippets across sessions |
| KAI-PERSONA-015 | HIGH | Self-reflection treats anonymous feedback and self-created corrections as performance truth |
| KAI-PERSONA-016 | HIGH | Self-reflection includes poisoned and cross-user correction records |
| KAI-PERSONA-017 | HIGH | Synthetic reflection conclusions are persisted and reused as behavioural evidence |
| KAI-PERSONA-018 | HIGH | Relationship metrics aggregate every user and system-generated memory |
| KAI-PERSONA-019 | HIGH | “Keeper memories” are counted by the pin flag rather than authenticated keeper identity |
| KAI-PERSONA-020 | HIGH | Relationship age is derived from caller-controlled memory timestamps |
| KAI-PERSONA-021 | HIGH | Arbitrary relationship milestones become persistent identity and graph evidence |
| KAI-PERSONA-022 | HIGH | Domain confidence is an uncalibrated count heuristic rather than measured correctness |
| KAI-PERSONA-023 | HIGH | Fabricated feedback affects confidence both through memories and the feedback store |
| KAI-PERSONA-024 | HIGH | Correction counts can exceed the denominator used for error rate |
| KAI-PERSONA-025 | HIGH | Confidence ignores sample independence, age, source quality and task difficulty |
| KAI-PERSONA-026 | HIGH | Any caller can submit a “correction” and trigger the confession engine |
| KAI-PERSONA-027 | HIGH | Confession output discloses private same-category advice merely because it is related |
| KAI-PERSONA-028 | HIGH | Caller-selected categories can activate confession cooldowns and suppress legitimate warnings |
| KAI-PERSONA-029 | HIGH | Autobiography text and context are stored without normal sanitisation or size bounds |
| KAI-PERSONA-030 | HIGH | Autobiographical significance is gameable through keywords and message length |
| KAI-PERSONA-031 | HIGH | Autobiography/milestone graph writes are untracked and not part of persistence success |
| KAI-PERSONA-032 | HIGH | Identity narrative combines all users, synthetic records and anonymous feedback into one self-model |
| KAI-PERSONA-033 | HIGH | A forged early timestamp changes Kai’s claimed age and relationship history |
| KAI-PERSONA-034 | HIGH | Arbitrary correction-event records become claimed autobiographical learning |
| KAI-PERSONA-035 | HIGH | Story arcs sort unvalidated timestamp strings lexicographically |
| KAI-PERSONA-036 | HIGH | Story arcs are driven by caller-manipulable correction/category counts |
| KAI-PERSONA-037 | HIGH | Absence of correction is labelled mastery without evidence of accuracy |
| KAI-PERSONA-038 | HIGH | Future-self improvement times are fixed constants presented as projections |
| KAI-PERSONA-039 | HIGH | Goal progress is inferred as ten percent per progress-note entry |
| KAI-PERSONA-040 | HIGH | Invalid goal timestamps are treated as one-day-old and inflate progress velocity |
| KAI-PERSONA-041 | HIGH | Overall trajectory uses unverified memory-event ratios as an error rate |
| KAI-PERSONA-042 | HIGH | Legacy messages accept unbounded raw content |
| KAI-PERSONA-043 | HIGH | Legacy recipient is an unrestricted caller string |
| KAI-PERSONA-044 | HIGH | Legacy delay accepts unbounded and non-finite values without a stable schema |
| KAI-PERSONA-045 | HIGH | Legacy IDs can collide under concurrent same-second writes |
| KAI-PERSONA-046 | HIGH | `include_unsurfaced=true` exposes private future time-capsule messages immediately |
| KAI-PERSONA-047 | HIGH | Reading legacy messages mutates their surfaced state |
| KAI-PERSONA-048 | HIGH | Legacy surfaced-state update is a non-atomic list scan followed by LSET |
| KAI-PERSONA-049 | HIGH | Counterfactual replay performs no simulation or outcome comparison |
| KAI-PERSONA-050 | HIGH | Counterfactuals are grounded in cross-user memory retrieval |
| KAI-PERSONA-051 | HIGH | Counterfactual original text is unbounded and not sanitised before search/storage |
| KAI-PERSONA-052 | HIGH | Poisoned or synthetic corrections and reflections drive counterfactual “lessons” |
| KAI-PERSONA-053 | HIGH | Any caller can overwrite the single global empathy map |
| KAI-PERSONA-054 | HIGH | Surface keywords are presented as inferred unspoken emotional needs |
| KAI-PERSONA-055 | HIGH | The global empathy map is disclosed to every caller and injected into Agentic context |
| KAI-PERSONA-056 | HIGH | Creative synthesis mixes memory snippets from all users and domains |
| KAI-PERSONA-057 | HIGH | Creative outputs embed private source snippets in the returned “connection” prompt |
| KAI-PERSONA-058 | HIGH | Random domain pairing and exact-string overlap are mislabeled creativity and novelty |
| KAI-PERSONA-059 | HIGH | Any caller can fabricate Kai’s inner thoughts and their context |
| KAI-PERSONA-060 | HIGH | The complete inner-monologue stream is externally readable |
| KAI-PERSONA-061 | HIGH | Any caller can create persistent aspirations and choose their domain |
| KAI-PERSONA-062 | HIGH | Aspiration feasibility uses memory volume as learning velocity |
| KAI-PERSONA-063 | HIGH | Invalid timestamps become epoch zero in aspiration/history calculations |
| KAI-PERSONA-064 | HIGH | Value reinforcement is an unlocked read-modify-write and loses concurrent updates |
| KAI-PERSONA-065 | HIGH | Value entries retain caller experiences as supporting evidence without verification |
| KAI-PERSONA-066 | HIGH | Formed values, strengths and supporting experiences are disclosed globally |
| KAI-PERSONA-067 | HIGH | Conscience alignment is keyword substring matching that ignores negation, intent and context |
| KAI-PERSONA-068 | HIGH | Conscience scoring counts matches equally and ignores learned value strength |
| KAI-PERSONA-069 | HIGH | No-match checks create neutral audit entries and an empty audit defaults to perfect integrity |
| KAI-PERSONA-070 | HIGH | Reading the conscience audit mutates the running alignment and streak |
| KAI-PERSONA-071 | HIGH | Repeated audit reads can indefinitely increase the alignment streak |
| KAI-PERSONA-072 | HIGH | Any caller can record loyalty for any person and it is automatically marked honoured |
| KAI-PERSONA-073 | HIGH | Sacrifice keywords automatically elevate loyalty weight to maximum |
| KAI-PERSONA-074 | HIGH | Gratitude text can automatically create maximum-weight loyalty records |
| KAI-PERSONA-075 | HIGH | Conscience, loyalty and gratitude stores merge all callers into one moral identity |
| KAI-PERSONA-076 | HIGH | Redis failure splits conscience state between workers and the shared store |
| KAI-PERSONA-077 | HIGH | Scheduled-task payload accepts arbitrary nested data and is disclosed to readers |
| KAI-PERSONA-078 | HIGH | Due scheduling compares unvalidated ISO text lexicographically |
| KAI-PERSONA-079 | HIGH | Monthly recurrence is implemented as a fixed 28-day interval |
| KAI-PERSONA-080 | HIGH | Task and reminder fire/cancel counters use non-atomic read-modify-write updates |
| KAI-PERSONA-081 | HIGH | All task and reminder content/payloads are globally enumerable |
| KAI-PERSONA-082 | HIGH | Any caller can mark tasks/reminders fired or cancelled and suppress delivery |
| KAI-PERSONA-083 | HIGH | Supervisor marks an item fired without requiring successful Telegram delivery status |
| KAI-PERSONA-084 | HIGH | Failure to mark a delivered item can cause repeated Telegram delivery |
| KAI-PERSONA-085 | HIGH | Morning/evening briefings combine goals, reminders, emotions and memories from all users |
| KAI-PERSONA-086 | HIGH | “Last 24h” emotion briefing uses the last 24 entries rather than a time window |
| KAI-PERSONA-087 | HIGH | Evening “interactions today” counts every memory event, including background/system records |
| KAI-PERSONA-088 | HIGH | The action registry exposes internal endpoints and may advertise stale/nonexistent actions |
| KAI-PERSONA-089 | HIGH | Echo analysis links unrelated sessions solely by a shared keyword-derived emotion |
| KAI-PERSONA-090 | HIGH | Cross-mode scanning searches and returns content from every user’s memory |
| KAI-PERSONA-091 | HIGH | PUB/WORK mode is inferred from stereotyped keywords and weak word overlap |
| KAI-PERSONA-092 | HIGH | Oracle impact prediction invents causal consequences from one-word overlap and generic verbs |
| KAI-PERSONA-093 | HIGH | Shadow branches are narrative templates that copy private memories, not simulated alternate timelines |
| KAI-PERSONA-094 | HIGH | Operator-model completeness rewards mere record presence rather than accuracy, consent or calibration |
| KAI-PERSONA-095 | MEDIUM | DND hours accept negative, non-finite and extreme values |
| KAI-PERSONA-096 | MEDIUM | Any caller can globally suppress proactive messaging through DND |
| KAI-PERSONA-097 | MEDIUM | “Critical” urgency bypasses DND without a trusted urgency authority |
| KAI-PERSONA-098 | MEDIUM | Arbitrary nudge types and repeated dismissals can grow counters and overflow exponentiation |
| KAI-PERSONA-099 | MEDIUM | DND, dismissals, cooldown and last-sent state are process-local and restart-volatile |
| KAI-PERSONA-100 | MEDIUM | Active-topic text and context are unbounded and unsanitised |
| KAI-PERSONA-101 | MEDIUM | Substring fuzzy matching merges unrelated topics |
| KAI-PERSONA-102 | MEDIUM | Deferred-topic storage is unbounded, process-local and globally shared |
| KAI-PERSONA-103 | MEDIUM | Deferred resurface delay accepts negative, non-finite and extreme values |
| KAI-PERSONA-104 | MEDIUM | Due deferred topics repeat forever until a separate mutable call marks them resurfaced |
| KAI-PERSONA-105 | MEDIUM | Unknown proactive modes silently receive the more permissive PUB configuration |
| KAI-PERSONA-106 | MEDIUM | Reading filtered proactive nudges mutates the last-sent cooldown state |
| KAI-PERSONA-107 | MEDIUM | Greeting time-of-day uses the service host timezone rather than the operator timezone |
| KAI-PERSONA-108 | MEDIUM | Check-in silence duration is based on process startup, not last operator interaction |
| KAI-PERSONA-109 | MEDIUM | Check-in subtracts string timestamps from floats, suppressing activity logic through exceptions |
| KAI-PERSONA-110 | MEDIUM | Full proactive scan suppresses every component failure and still returns `status: ok` |
| KAI-PERSONA-111 | MEDIUM | Proactive urgency values are fixed constants rather than derived/calibrated risk |
| KAI-PERSONA-112 | MEDIUM | Fading-memory nudges depend on manipulable access counts and flawed recency calculations |
| KAI-PERSONA-113 | MEDIUM | Briefing generation can be spammed and appends a new history record on every call |
| KAI-PERSONA-114 | MEDIUM | Reminder repeat validation accepts frequencies beyond the endpoint’s documented contract |
| KAI-PERSONA-115 | MEDIUM | Task and reminder IDs use only the first 12 UUID characters |
| KAI-PERSONA-116 | MEDIUM | Capacity checks and inserts are separate operations and can exceed configured caps |
| KAI-PERSONA-117 | MEDIUM | Redis RPUSH and LTRIM are separate operations, not one atomic capped append |
| KAI-PERSONA-118 | MEDIUM | Redis list entry updates race with concurrent trimming/insertion |
| KAI-PERSONA-119 | MEDIUM | Empathy-map GET/update/SET is a non-atomic read-modify-write |
| KAI-PERSONA-120 | MEDIUM | Alignment HSET, streak increment and readback are not one atomic transaction |
| KAI-PERSONA-121 | MEDIUM | P21 item updates are explicitly non-atomic read-modify-write operations |
| KAI-PERSONA-122 | MEDIUM | Nudge-ladder eviction and replacement can race concurrent updates |
| KAI-PERSONA-123 | MEDIUM | Redis errors silently fall back to independent in-memory personality states |
| KAI-PERSONA-124 | MEDIUM | Corrupt Redis JSON is silently replaced with fallback state |
| KAI-PERSONA-125 | MEDIUM | P17–P22 periodic persist/restore functions are operational no-ops |
| KAI-PERSONA-126 | MEDIUM | Multiple list endpoints accept negative or unbounded limits with surprising slicing semantics |
| KAI-PERSONA-127 | MEDIUM | Identity, projection, imagination and briefing calls perform repeated 10,000-record scans |
| KAI-PERSONA-128 | MEDIUM | Synchronous Redis, database and embedding work runs in async handlers |
| KAI-PERSONA-129 | MEDIUM | P17–P22 state has no authenticated user, consent, timezone or retention partition |
| KAI-PERSONA-130 | MEDIUM | Persistence and delivery records lack an immutable causal link from source input to derived state/output |

---

## Critical evidence-poisoning paths

### KAI-PERSONA-001 — CRITICAL — Feedback becomes authoritative memory
`POST /memory/feedback` accepts any session ID/index/rating. Ratings 4–5 create `feedback_positive` memories at importance 0.85; ratings 1–2 create `correction` memories at importance 0.90. These records directly alter domain confidence, identity, story arcs, reflection, planning and counterfactuals. There is no authenticated operator or message ownership proof.

### KAI-PERSONA-002 — CRITICAL — Moral-model and integrity forgery
`POST /memory/values/learn` accepts arbitrary experience/outcome and defaults every non-`positive`/`negative` outcome to the positive signal set. `POST /memory/conscience/check` then writes caller-selected actions/verdicts into the audit. Agentic reads formed values and integrity into privileged context. Anonymous input can therefore create the moral model and its evidence of compliance.

---

## Externally delivered proactive state

### KAI-PERSONA-003 — HIGH — Reminder-to-Telegram injection
Supervisor polls `/memory/reminders/due` and sends each reminder text to `TELEGRAM_ALERT_URL`. Any caller can create the reminder.

### KAI-PERSONA-004 — HIGH — Scheduled-title Telegram injection
Supervisor polls `/memory/schedule/due` and sends attacker-controlled task titles as scheduled messages.

### KAI-PERSONA-005 — HIGH — Escalation/harassment injection
Repeated calls to `/memory/nudge/escalate` reach tough-love/intervention tiers. Supervisor sends level 3/4 targets to Telegram as escalated nudges.

---

## P16/P17 feedback, emotion and relationship findings

### KAI-PERSONA-006 — HIGH — No feedback ownership
The endpoint does not establish that the caller owns the session, that the selected message is an assistant response or that the rating came from the operator.

### KAI-PERSONA-007 — HIGH — Unsafe negative indexing
The check only tests `message_index < len(msgs)`. `-1` selects the latest entry; larger negative values raise inside a swallowed exception.

### KAI-PERSONA-008 — HIGH — Unsanitised correction comment
The feedback entry uses a sanitised comment, but the correction memory stores `req.comment` directly.

### KAI-PERSONA-009 — HIGH — Volatile global feedback
Feedback exists only in one process list, merges all sessions/users and disappears on restart.

### KAI-PERSONA-010 — HIGH — False feedback success
Memory creation exceptions are swallowed, but the response still reports `effect: boost` or `correction`.

### KAI-PERSONA-011 — HIGH — Feedback disclosure
Statistics include recent entries with session IDs, comments, message indices and ratings.

### KAI-PERSONA-012 — HIGH — Weak emotion semantics
Emotion is the maximum count of substrings. Negation, quotation, subject, sarcasm and context are ignored.

### KAI-PERSONA-013 — HIGH — Cross-caller emotional identity
All recorded emotional states feed one timeline and one claimed operator mood.

### KAI-PERSONA-014 — HIGH — Emotional snippet disclosure
Timeline and echo responses expose stored trigger text and timestamps from other sessions.

### KAI-PERSONA-015 — HIGH — Self-review trusts poisoned inputs
Reflection uses global feedback counts, emotion counts and correction records as though they were verified outcomes.

### KAI-PERSONA-016 — HIGH — Poison/quarantine ignored
The correction scan filters by event type but not `poisoned`, user ID, trust tier or source.

### KAI-PERSONA-017 — HIGH — Recursive synthetic evidence
Generated reflections are persisted and later reused by identity and counterfactual modules as strengths/weaknesses.

### KAI-PERSONA-018 — HIGH — Global relationship narrative
Days together, categories, corrections, ratings, emotions and milestones aggregate every caller and background record.

### KAI-PERSONA-019 — HIGH — Incorrect keeper metric
`keeper_count` counts `content.pin`, not memories created by or verified as the keeper.

### KAI-PERSONA-020 — HIGH — Forged relationship age
The earliest unvalidated memory timestamp defines days together.

### KAI-PERSONA-021 — HIGH — Milestone poisoning
Any text becomes a relationship milestone and is asynchronously propagated to the graph.

### KAI-PERSONA-022 — HIGH — Confidence is not calibrated correctness
Confidence is derived from total event counts and correction ratios, not verified predictions/outcomes.

### KAI-PERSONA-023 — HIGH — Double influence from feedback
A rating can both enter `_feedback_store` and create a correction/positive memory, influencing confidence/reflection twice.

### KAI-PERSONA-024 — HIGH — Invalid denominator
Feedback-store corrections increase `corrections` but not `total`; correction count can exceed total and force the minimum score.

### KAI-PERSONA-025 — HIGH — No evidence quality model
Source independence, age, task difficulty, sample size and correctness review are absent.

### KAI-PERSONA-026 — HIGH — Caller-created confession event
Any text labelled correction starts the confession search; no actual corrected claim or outcome link is required.

### KAI-PERSONA-027 — HIGH — Private advice disclosure by association
Related keeper memories in the same heuristic category are returned as “potentially wrong” original advice.

### KAI-PERSONA-028 — HIGH — Cooldown suppression
Caller-selected categories become shared confession-cooldown keys and can suppress later legitimate warnings.

---

## P18 narrative-identity findings

### KAI-PERSONA-029 — HIGH — Raw autobiography storage
Autobiography uses raw `.strip()` text/context before storing snippets and generating graph records; aggregate input bounds are absent.

### KAI-PERSONA-030 — HIGH — Significance gaming
Keywords such as `breakthrough`, `wrong`, `love` or simply text over 500 characters cross the autobiography threshold.

### KAI-PERSONA-031 — HIGH — Unacknowledged graph fan-out
Autobiography and milestones create untracked graph tasks whose success is not tied to the API result.

### KAI-PERSONA-032 — HIGH — Cross-user self-narrative
Identity combines all memory categories, corrections, emotional records and reflections into one first-person narrative.

### KAI-PERSONA-033 — HIGH — Age poisoning
Lexically smallest caller timestamp can redefine the claimed start date.

### KAI-PERSONA-034 — HIGH — Unverified learning claims
Every record whose event type equals `correction` becomes something Kai claims it learned from.

### KAI-PERSONA-035 — HIGH — Lexical story chronology
Story arcs sort timestamp strings rather than canonical instants.

### KAI-PERSONA-036 — HIGH — Manipulable chapter classification
Correction/category records decide whether a chapter is mastery, expansion or growing pains.

### KAI-PERSONA-037 — HIGH — Mastery from missing criticism
A correction rate below 5% is labelled mastery without positive evidence or test performance.

### KAI-PERSONA-038 — HIGH — Fixed future claims
Low/medium/high domain confidence maps to 30/14/0 estimated days with no learned model.

### KAI-PERSONA-039 — HIGH — Progress-note arithmetic
Every goal progress note is treated as ten percentage points.

### KAI-PERSONA-040 — HIGH — Invalid-time velocity inflation
Malformed goal timestamps default age to one day, maximising apparent progress per day.

### KAI-PERSONA-041 — HIGH — Synthetic trajectory
Memory-event proportions are presented as learning/error trajectory despite unverified sources and background records.

### KAI-PERSONA-042 — HIGH — Unbounded legacy content
Legacy messages are not sanitised/truncated before Redis/list storage and response.

### KAI-PERSONA-043 — HIGH — Unvalidated recipient
Any recipient string is accepted despite the documented `self|operator` contract.

### KAI-PERSONA-044 — HIGH — Unsafe delay
Only values below one are clamped. Huge, non-finite or non-numeric values can fail or create unusable schedules.

### KAI-PERSONA-045 — HIGH — Weak legacy identity
ID uses current integer second plus list length, which races concurrent writers.

### KAI-PERSONA-046 — HIGH — Time-capsule confidentiality bypass
The query flag returns all future messages before their intended surface date.

### KAI-PERSONA-047 — HIGH — Read mutates state
GET marks ready messages surfaced, mixing retrieval and irreversible state change.

### KAI-PERSONA-048 — HIGH — Lost surfaced-state updates
List scan and `LSET` are not atomic against concurrent append/trim.

---

## P19 imagination findings

### KAI-PERSONA-049 — HIGH — Counterfactual branding mismatch
The output is a deterministic set of keyword observations and generic text, not a simulated alternative with predicted outcomes or comparison.

### KAI-PERSONA-050 — HIGH — Cross-user grounding
Related-memory search has no user scope.

### KAI-PERSONA-051 — HIGH — Raw original text
`original` is searched before sanitisation/truncation and can be arbitrarily large.

### KAI-PERSONA-052 — HIGH — Poisoned lesson chain
Synthetic reflections and correction-event records determine known weaknesses and lessons without provenance validation.

### KAI-PERSONA-053 — HIGH — Global empathy overwrite
Every empathise call replaces the same current operator state.

### KAI-PERSONA-054 — HIGH — Unsupported mind-reading
Message length, punctuation and keyword lists produce claims about unspoken needs and communication style.

### KAI-PERSONA-055 — HIGH — Empathy disclosure and prompt injection
The shared map is externally readable and Agentic injects it as privileged context.

### KAI-PERSONA-056 — HIGH — Cross-user creative source pool
Creative synthesis groups all non-poisoned memory text by category without user/trust partition.

### KAI-PERSONA-057 — HIGH — Source text leakage
Returned connection strings embed memory snippets from both selected domains.

### KAI-PERSONA-058 — HIGH — False novelty score
Random pairing drives output; novelty is one minus exact-string set overlap, not novelty, utility or correctness.

### KAI-PERSONA-059 — HIGH — Fabricated inner thought
Any caller can store text as “what Kai is really thinking”.

### KAI-PERSONA-060 — HIGH — Inner-state disclosure
The complete thought stream and distribution are readable by all callers.

### KAI-PERSONA-061 — HIGH — Aspiration poisoning
Any caller creates Kai’s stated future vision and domain.

### KAI-PERSONA-062 — HIGH — Volume as learning
Feasibility cites memories per day as learning velocity, irrespective of correctness, duplication or source.

### KAI-PERSONA-063 — HIGH — Invalid timestamps silently become zero
`_parse_ts()` returns epoch zero, silently excluding malformed records from recent learning.

---

## P20 conscience/value findings

### KAI-PERSONA-064 — HIGH — Lost reinforcement updates
Value lookup, strength increment and write are separate operations.

### KAI-PERSONA-065 — HIGH — Unverified supporting experiences
Caller text is stored in the value’s experience list and presented as formation evidence.

### KAI-PERSONA-066 — HIGH — Moral-profile disclosure
All values, strengths, experiences and alignment state are globally returned.

### KAI-PERSONA-067 — HIGH — Weak conscience semantics
Positive/negative keyword presence decides alignment and ignores negation, target and consequences.

### KAI-PERSONA-068 — HIGH — Strength ignored
Overall alignment is based on number of matching values, not their learned strengths.

### KAI-PERSONA-069 — HIGH — Perfect-by-default integrity
No matches create neutral audit records; an empty audit reports integrity 1.0.

### KAI-PERSONA-070 — HIGH — Audit read changes governance state
GET recalculates and writes running alignment, violations and streak.

### KAI-PERSONA-071 — HIGH — Streak inflation by polling
Every read with no conflicts increments the streak again, including the same unchanged log.

### KAI-PERSONA-072 — HIGH — Forged loyalty
Caller-selected person/act/type is stored with `honored=True` without evidence.

### KAI-PERSONA-073 — HIGH — Keyword maximum-weight promotion
Sacrifice phrases automatically set type and weight to maximum.

### KAI-PERSONA-074 — HIGH — Gratitude-to-loyalty side effect
A gratitude reason matching sacrifice keywords creates a second maximum-weight loyalty record.

### KAI-PERSONA-075 — HIGH — Global moral identity
No user/source partition exists across values, conscience, loyalty or gratitude.

### KAI-PERSONA-076 — HIGH — Failover divergence
Redis failures switch individual calls to per-process lists/maps; later recovery does not reconcile them.

---

## P21/P22 proactive and operator-model findings

### KAI-PERSONA-077 — HIGH — Arbitrary task payload
Scheduled task payload is an unrestricted nested dictionary retained and returned.

### KAI-PERSONA-078 — HIGH — Lexical due checks
Unvalidated `fire_at` text is compared with ISO strings; malformed/timezone-varied values fire incorrectly.

### KAI-PERSONA-079 — HIGH — Incorrect monthly recurrence
Monthly means 28 elapsed days, not a calendar month or requested local schedule.

### KAI-PERSONA-080 — HIGH — Lost counter updates
Task/reminder fire counts and active/fired state use non-atomic read-modify-write.

### KAI-PERSONA-081 — HIGH — Schedule disclosure
List/due/briefing endpoints reveal every task/reminder text, payload and timing.

### KAI-PERSONA-082 — HIGH — Delivery suppression
Anonymous fire/cancel calls can hide tasks/reminders before Supervisor delivers them.

### KAI-PERSONA-083 — HIGH — Delivery status not verified
Supervisor calls Telegram and then marks the item fired without checking the Telegram response status.

### KAI-PERSONA-084 — HIGH — Duplicate delivery window
If Telegram succeeds but the memU fire request fails, the item remains due and is sent again.

### KAI-PERSONA-085 — HIGH — Cross-user briefings
Goals, reminders, emotional state, nudges and activity are not partitioned.

### KAI-PERSONA-086 — HIGH — Wrong emotion window
The last 24 entries are labelled yesterday/24h irrespective of timestamps.

### KAI-PERSONA-087 — HIGH — Memory events called interactions
Evening activity counts all records beginning with today’s date.

### KAI-PERSONA-088 — HIGH — Static action authority
The registry exposes internal route topology and can advertise actions whose implementation, gating or method has drifted.

### KAI-PERSONA-089 — HIGH — Cross-session emotional echo
Same keyword-derived emotion plus a different caller-selected session ID is enough to quote a past trigger.

### KAI-PERSONA-090 — HIGH — Cross-mode exfiltration
The scan reads all memory content and returns opposite-mode snippets without user scope.

### KAI-PERSONA-091 — HIGH — Stereotyped mode inference
Words such as `beer`, `girl`, `mate`, `invoice` and `site` infer mode; simple overlap determines relevance.

### KAI-PERSONA-092 — HIGH — Fabricated causal forecast
One overlapping word links an action to a goal; generic verbs such as `do/start/skip` determine positive/negative impact and risk narrative.

### KAI-PERSONA-093 — HIGH — Shadow narrative, not simulation
The branch copies related memory snippets and outputs a template saying trajectory would change; it estimates no outcome probability or causal path.

### KAI-PERSONA-094 — HIGH — Completeness without correctness
Operator-model completeness awards 20 points for any data in each subsystem, including attacker-created records, and ignores accuracy/consent/freshness.

---

## Medium-severity operational/state findings

### KAI-PERSONA-095 — MEDIUM — DND numeric validation
Negative, NaN, infinity and huge hours are accepted into wall-clock arithmetic.

### KAI-PERSONA-096 — MEDIUM — Global DND control
Any caller can suppress operator nudges for the process.

### KAI-PERSONA-097 — MEDIUM — Untrusted urgency override
Any nudge labelled urgency at least 0.9 bypasses DND.

### KAI-PERSONA-098 — MEDIUM — Dismissal counter abuse
Arbitrary type strings create state; repeated calls can drive expensive exponentiation before the cooldown cap.

### KAI-PERSONA-099 — MEDIUM — Volatile anti-annoyance state
DND/cooldowns/dismissals are worker-local and reset on restart.

### KAI-PERSONA-100 — MEDIUM — Topic size/sanitisation
Topic/context fields have no length or sanitisation constraints.

### KAI-PERSONA-101 — MEDIUM — Topic collision
Containment matching merges distinct topics when one phrase is a substring of another.

### KAI-PERSONA-102 — MEDIUM — Deferred-topic growth
The deferred list has no cap, persistence or user partition.

### KAI-PERSONA-103 — MEDIUM — Invalid defer timing
Negative/non-finite values make topics immediately due or create invalid timestamps.

### KAI-PERSONA-104 — MEDIUM — Repeated resurfacing
Listing due topics does not mark them surfaced; they repeat until a separate call succeeds.

### KAI-PERSONA-105 — MEDIUM — Unknown mode becomes PUB
Invalid modes use the more permissive PUB nudge set.

### KAI-PERSONA-106 — MEDIUM — Read changes cooldown
Filtered proactive retrieval marks selected nudge types as sent even if nothing is delivered.

### KAI-PERSONA-107 — MEDIUM — Wrong timezone greeting
Time-of-day uses host-local `datetime.now()` rather than operator-configured timezone.

### KAI-PERSONA-108 — MEDIUM — False silence duration
Check-in uses process startup time, not the last interaction.

### KAI-PERSONA-109 — MEDIUM — Check-in activity type error
It calculates `now - r.timestamp` where timestamps are normally strings; exceptions suppress the goal-progress message.

### KAI-PERSONA-110 — MEDIUM — Source failures hidden
Every component of the unified proactive scan is wrapped in broad exception suppression.

### KAI-PERSONA-111 — MEDIUM — Fixed urgency labels
Reminder/drift/fading urgency values are constants unrelated to verified deadline, harm or operator priority.

### KAI-PERSONA-112 — MEDIUM — Manipulable fading logic
Access-count/stability and timestamp defects control whether a memory is described as important and fading.

### KAI-PERSONA-113 — MEDIUM — Briefing history spam
Every morning/evening call appends a new briefing; no daily idempotency or actor check exists.

### KAI-PERSONA-114 — MEDIUM — Repeat contract drift
Reminders accept the broader task frequency set, including hourly/monthly, despite narrower endpoint documentation.

### KAI-PERSONA-115 — MEDIUM — Short IDs
Task/reminder IDs expose only 48 UUID bits, reducing collision resistance under persistent distributed use.

### KAI-PERSONA-116 — MEDIUM — Capacity race
Length check and HSET are separate, so concurrent creators can exceed the cap.

### KAI-PERSONA-117 — MEDIUM — Non-atomic capped append
RPUSH and LTRIM are individually atomic but not one transaction; crashes/concurrency can temporarily or permanently violate cap expectations.

### KAI-PERSONA-118 — MEDIUM — List-update race
Legacy/list updates use index positions that may change between LRANGE and LSET.

### KAI-PERSONA-119 — MEDIUM — Empathy-map race
GET, dictionary update and SET can overwrite concurrent fields.

### KAI-PERSONA-120 — MEDIUM — Alignment transaction gap
Overall/violations HSET, streak HINCRBY and readback can interleave.

### KAI-PERSONA-121 — MEDIUM — P21 item race
The code explicitly performs read-modify-write on a JSON hash field without CAS/Lua/transaction.

### KAI-PERSONA-122 — MEDIUM — Ladder eviction race
Capacity inspection, oldest deletion, state update and write are separate operations.

### KAI-PERSONA-123 — MEDIUM — Redis split-brain fallback
Every helper silently falls back to module state on Redis failure; workers diverge and later success does not merge records.

### KAI-PERSONA-124 — MEDIUM — Corruption hidden
JSON decode/Redis type errors are swallowed and fallback state is presented normally.

### KAI-PERSONA-125 — MEDIUM — No-op persistence lifecycle
Periodic persist and startup restore return empty result maps because all P17–P22 sections are skipped.

### KAI-PERSONA-126 — MEDIUM — Weak list limits
Several list APIs use ordinary integer slicing; negative or huge limits have surprising/unbounded effects.

### KAI-PERSONA-127 — MEDIUM — Repeated full-store scans
Narrative, future, imagination, confidence and briefing endpoints repeatedly load up to 10,000 records.

### KAI-PERSONA-128 — MEDIUM — Blocking async execution
Redis client, database search and embedding operations are synchronous inside async routes.

### KAI-PERSONA-129 — MEDIUM — Missing identity/consent partition
These highly personal state models have no authenticated user, consent revision, timezone, retention or deletion boundary.

### KAI-PERSONA-130 — MEDIUM — Missing causal provenance
Derived feelings, values, predictions, nudges and identity statements do not retain a tamper-evident link to the exact source record, actor and algorithm revision.

---

## Batch totals

- Findings: **130**
- Critical: **2**
- High: **92**
- Medium: **36**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,452**
- Critical: **122**
- High: **670**
- Medium: **657**
- Low: **3**

## Files materially reviewed

P17–P22 implementations and Redis helpers in `memu-core/app.py`, plus external reminder/task/escalation delivery in `supervisor/app.py`.
