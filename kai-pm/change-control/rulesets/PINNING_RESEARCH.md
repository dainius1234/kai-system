# CAI v1.0 — GitHub capability research (order §20) and the identity finding

Measured 2026-09-23. Every claim names its source. Where a fact can only
be established by a live server-side change, it is marked **UNVERIFIED**:
leg 1 forbids live ruleset changes, so it was not attempted.

## 0. Sources

| source | how obtained | identity |
|---|---|---|
| GitHub REST OpenAPI description, `api.github.com.json` | `raw.githubusercontent.com/github/rest-api-description/main/descriptions/api.github.com/api.github.com.json` | 12,964,474 bytes; sha256 `31017b9d3820df4ef4fbef8bfb72a86045687758b91c45896ef53021c6f2cd5a`; `info.version` 1.1.4; ETag `b7807b18…ffb0`; fetched 2026-09-23T10:14:40Z |
| repository metadata | `GET https://api.github.com/repos/dainius1234/kai-system` (unauthenticated) | live |
| live rulesets / branch rules | `GET …/rulesets`, `GET …/rules/branches/main` (unauthenticated) | live |
| push identity | `GET …/events` (public event log) and the session's GitHub connector `get_me` | live |

`docs.github.com` was **not** reachable: the session's network policy
blocks that host. The OpenAPI description is the machine-readable
contract the documentation is generated from, and it was used instead.

## 1. Repository facts

* `id` **1004463473**, `full_name` `dainius1234/kai-system` — both match
  the implementation order and the payload binding.
* `private: false`, `visibility: public`, owner type **User** (a personal
  account, not an organisation).
* live rulesets: `[]`; rules on `main`: `[]`. CAI enforces nothing today.

## 2. What the ruleset API accepts (schema facts)

* `POST /repos/{owner}/{repo}/rulesets` accepts `target` ∈ `branch`,
  `tag`, `push` and 25 rule types, **including `workflows`**.
* `workflows` rule → `WorkflowFileReference {path, repository_id, ref,
  sha}`; `sha` is described as "The commit SHA of the workflow file to
  use". **Exact-SHA pinning is expressible.**
* `required_status_checks` → `{context, integration_id?}`; a context is a
  name. Naming a check is not pinning a verifier (order §20), so the
  design uses `workflows` + `sha`, never a check name.
* `creation`, `update` (`update_allows_fetch_and_merge`), `deletion`
  apply to tag rulesets; bypass actor types include `User`,
  `RepositoryRole`, `Integration`, `DeployKey`, `Team`;
  **`OrganizationAdmin` "is not applicable for personal repositories"**.
* `enforcement` ∈ `disabled`, `active`, `evaluate`, and **`evaluate` "is
  only available with GitHub Enterprise"**. Leg 2 therefore cannot trial
  the rules in evaluate mode; they go straight to `active`.
* The `pull_request` rule's five required parameters were checked against
  the schema; the proposed file carries exactly those five.

## 3. **UNVERIFIED — returned to Kai**: required-workflow availability

The schema a repository-level create accepts includes `workflows`. The
schema is one document for every plan, so it cannot say whether GitHub's
**server** accepts that rule for a **User-owned** repository on this
account's plan. Only a server-side create answers that, and leg 1 forbids
live ruleset changes. Per order §20 this part STOPS here: the design uses
exact-SHA `workflows` pinning and does **not** fall back to a check name.
If leg 2 finds the rule unavailable, that is a capability gap for Kai, not
a substitution for Orion.

## 4. **ARCHITECTURE FINDING — returned to Kai and Dainius**: identity collapse

**The executor pushes as Dainius.** Source: GitHub's own event log for
this repository records **every** push in this session's lineage — the
CAI branch (`5ccd14d`, and the branch creation) and every D379 commit
checked (`5b20447` … `0af5d32`) — with `actor = dainius1234 (216391246)`.
The session's GitHub connector also authenticates as `dainius1234`
(216391246).

To GitHub, **Orion and Dainius are one identity.** Consequences, derived
from the ruleset semantics in §2:

1. A bypass actor that is Dainius is a bypass for Orion. The creation
   rule in `10-authority-tags-create.json` separates nothing.
2. As the same principal, the executor holds repository-owner powers,
   including deleting or editing rulesets themselves.
3. **Deleting the head admission tag silently regresses the programme
   baseline.** The remaining chain is still complete and valid, and the
   verifier as designed cannot see what is missing. Immutability rulesets
   are the intended defence, and (2) defeats them.
4. The threat model's in-scope "compromised executor credentials" is,
   today, the out-of-scope "fully compromised Dainius GitHub account".

What **still** holds: the signature pin. The executor cannot *forge* an
authoritative record, because Dainius's private signing key is not
available to it. Forgery is prevented; deletion and denial of service
are not.

Made mechanical, not left as prose: `check_cai_bootstrap.py
--apply-ready --executor-id <id>` **refuses** any design whose authority
bypass actor is the executor's identity (case B3; mutant M31). The
committed design carries a placeholder there, so it is not apply-ready.

Options for Kai / Dainius — **none implemented**, each changes identity
or architecture and is outside this tranche:

* a separate executor identity (machine account, GitHub App or
  fine-grained token) with write but **not** admin and **not** bypass,
  so rulesets can tell the two apart;
* an external anchor for the chain head, so deletion of the latest
  admission is detectable (for example, the verifier workflow's pinned
  revision carrying a minimum expected admission sequence).

## 5. Residuals stated

* Actions are pinned by tag (`actions/checkout@v4`,
  `actions/setup-python@v5`), matching this repository's measured
  convention (7 and 6 uses). Commit-SHA pinning was not possible from
  this session: the proxy enables GitHub access for this repository only.
* SSH-format tag signatures are not implemented (contract P-4).
