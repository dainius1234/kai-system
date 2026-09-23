"""CAI v1.0 hostile-calibration kit. Support module, no main().

Builds REAL worlds: a fresh Git repository, real OpenPGP keys, real signed
annotated tags. Nothing is mocked at the boundary under test: every case
invokes the SHIPPED checker as a subprocess and reads its real exit code
and output (doctrine rule 17 — the entry point is exercised directly).

Two keys exist: AUTHORITY (the key a case pins) and ATTACKER (any other
key). Tag creation happens HERE, in the test world, and never in the
verifiers — the verifiers create nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKERS = REPO_ROOT / "scripts" / "security"
REPO_ID = 1004463473
REPO_NAME = "dainius1234/kai-system"
FIXED_DATE = "1790000000 +0000"      # 2026-09-21T14:13:20Z, deterministic


def _env(extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_AUTHOR_NAME": "cai-test", "GIT_AUTHOR_EMAIL": "t@t",
                "GIT_COMMITTER_NAME": "cai-test", "GIT_COMMITTER_EMAIL": "t@t",
                "GIT_AUTHOR_DATE": FIXED_DATE, "GIT_COMMITTER_DATE": FIXED_DATE,
                "LC_ALL": "C"})
    env.update(extra or {})
    return env


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False) + "\n"


class Keys:
    """Two real ed25519 signing keys in a private signer home."""

    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="cai-keys-"))
        self.home = self.root / "signer"
        self.home.mkdir(mode=0o700)
        self.fpr = {}
        for name in ("AUTHORITY", "ATTACKER"):
            self._gpg("--quick-gen-key", f"{name} <{name.lower()}@cai.test>",
                      "ed25519", "sign", "never")
            out = self._gpg("--with-colons", "--fingerprint", "--list-keys",
                            f"{name.lower()}@cai.test").decode()
            self.fpr[name] = [l.split(":")[9] for l in out.splitlines()
                              if l.startswith("fpr:")][0]
        self.authority_pub = self.root / "authority.asc"
        self.authority_pub.write_bytes(self._gpg(
            "--armor", "--export", self.fpr["AUTHORITY"]))

    def _gpg(self, *args) -> bytes:
        pr = subprocess.run(["gpg", "--batch", "--homedir", str(self.home),
                             "--pinentry-mode", "loopback", "--passphrase", ""]
                            + list(args), capture_output=True, timeout=120)
        if pr.returncode != 0:
            raise RuntimeError(pr.stderr.decode()[:300])
        return pr.stdout

    def config(self, directory: Path, *, gpg_program="gpg",
               fingerprint="AUTHORITY") -> Path:
        cfg = {"schema": "kai.cai.authority_keys.v1", "issuer": "dainius1234",
               "pinned_primary_fingerprint":
                   self.fpr[fingerprint] if fingerprint else None,
               "public_key_armored_path": str(self.authority_pub),
               "gpg_program": gpg_program, "status": "TEST"}
        p = directory / f"authority_{fingerprint}_{Path(gpg_program).name}.json"
        p.write_text(json.dumps(cfg))
        return p

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


class World:
    """A fresh repository with a baseline commit B.

    B:  allowed/a.txt  allowed/b.txt  forbidden/x.txt  docs/readme.md
    """

    def __init__(self, keys: Keys):
        self.keys = keys
        self.dir = Path(tempfile.mkdtemp(prefix="cai-world-"))
        self.repo = self.dir / "repo"
        self.git("init", "-q", "-b", "main", str(self.repo), cwd=self.dir)
        for rel, txt in (("allowed/a.txt", "a\n"), ("allowed/b.txt", "b\n"),
                         ("forbidden/x.txt", "x\n"), ("docs/readme.md", "r\n")):
            self.write(rel, txt)
        self.B = self.commit("baseline B")

    # ── git ───────────────────────────────────────────────────────────
    def git(self, *args, cwd=None, env=None, input_bytes=None) -> str:
        pr = subprocess.run(["git"] + list(args), cwd=str(cwd or self.repo),
                            capture_output=True, env=_env(env),
                            input=input_bytes, timeout=120)
        if pr.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)}: "
                               f"{pr.stderr.decode()[:300]}")
        return pr.stdout.decode()

    def write(self, rel, txt):
        p = self.repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt)

    def commit(self, msg, parent_ref=None) -> str:
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", msg)
        return self.git("rev-parse", "HEAD").strip()

    def checkout(self, sha):
        self.git("checkout", "-q", "--detach", sha)

    def tree(self, sha) -> str:
        return self.git("rev-parse", f"{sha}^{{tree}}").strip()

    def same_tree_new_commit(self, sha, msg="same tree, different commit"):
        return self.git("commit-tree", self.tree(sha), "-p", sha, "-m",
                        msg).strip()

    # ── signed control records ────────────────────────────────────────
    def tag(self, name, target, payload, *, signer="AUTHORITY",
            raw_message=None) -> str:
        msgfile = self.dir / "msg.txt"
        msgfile.write_text(raw_message if raw_message is not None
                           else canonical(payload))
        base = ["-c", "gpg.format=openpgp", "-c", "gpg.program=gpg"]
        if signer is None:
            self.git(*base, "tag", "-a", "--cleanup=verbatim", "-F",
                     str(msgfile), name, target)
        else:
            self.git(*base, "-c", f"user.signingkey={self.keys.fpr[signer]}",
                     "tag", "-s", "--cleanup=verbatim", "-F", str(msgfile),
                     name, target,
                     env={"GNUPGHOME": str(self.keys.home)})
        return self.git("rev-parse", f"refs/tags/{name}").strip()

    def tampered_copy(self, src_ref, dst_ref, old: bytes, new: bytes) -> str:
        """Re-write a signed tag object's bytes keeping its signature, so
        the signature no longer covers what is signed (BADSIG)."""
        raw = subprocess.run(["git", "cat-file", "tag", src_ref],
                             cwd=self.repo, capture_output=True,
                             env=_env()).stdout
        assert raw.count(old) >= 1, "tamper target not found"
        raw = raw.replace(old, new, 1)
        sha = self.git("hash-object", "-t", "tag", "-w", "--stdin",
                       input_bytes=raw).strip()
        self.git("update-ref", dst_ref, sha)
        return sha

    def wa_payload(self, wa_id, baseline, **over):
        p = {"schema": "kai.cai.work_authority.v1", "repository_id": REPO_ID,
             "repository_full_name": REPO_NAME, "wa_id": wa_id,
             "baseline_commit": baseline, "baseline_tree": self.tree(baseline),
             "control_level": "C2", "allowed_paths": [],
             "allowed_prefixes": ["allowed/"], "forbidden_paths": [],
             "forbidden_prefixes": ["forbidden/"],
             "expires_at": "2030-01-01T00:00:00Z", "governing_refs": [],
             "required_verification": ["hostile calibration"],
             "prohibited_operations": ["merge to main"],
             "issuer": "dainius1234", "note": "test"}
        p.update(over)
        return p

    def issue_wa(self, wa_id, baseline, signer="AUTHORITY", **over) -> str:
        return self.tag(f"kai-wa/wa/{wa_id}", baseline,
                        self.wa_payload(wa_id, baseline, **over), signer=signer)

    def issue_event(self, wa_id, event_id, event, seq_at_issue=0,
                    signer="AUTHORITY") -> str:
        wa_obj = self.git("rev-parse", f"refs/tags/kai-wa/wa/{wa_id}").strip()
        return self.tag(f"kai-wa/ev/{wa_id}/{event_id}", wa_obj, {
            "schema": "kai.cai.terminal_event.v1", "repository_id": REPO_ID,
            "repository_full_name": REPO_NAME, "event_id": event_id,
            "wa_id": wa_id, "event": event,
            "admission_sequence_at_issue": seq_at_issue,
            "issuer": "dainius1234", "note": "test"}, signer=signer)

    def evidence(self, wa_id, candidate, *, disposition="ACCEPT_FOR_ADMISSION",
                 name="ivv.json"):
        """Commit an IV&V evidence file on a side commit; return
        (ref, sha256)."""
        here = self.git("rev-parse", "HEAD").strip()
        self.checkout(self.B)
        rel = f"ivv/{wa_id}-{candidate[:8]}-{name}"
        doc = {"schema": "kai.cai.ivv_evidence.v1", "repository_id": REPO_ID,
               "repository_full_name": REPO_NAME, "wa_id": wa_id,
               "candidate_commit": candidate,
               "candidate_tree": self.tree(candidate),
               "disposition": disposition, "reviewer": "Kai",
               "summary": "test evidence"}
        self.write(rel, canonical(doc))
        ev_commit = self.commit(f"evidence {rel}")
        self.checkout(here)
        data = (canonical(doc)).encode()
        return ({"kind": "git-blob", "commit": ev_commit, "path": rel},
                hashlib.sha256(data).hexdigest())

    def admit(self, adm_id, seq, wa_id, target, *, candidate=None, ev=None,
              prev=None, signer="AUTHORITY", **over) -> str:
        candidate = candidate or target
        if ev is None:
            ev = self.evidence(wa_id, candidate)
        p = {"schema": "kai.cai.admission.v1", "repository_id": REPO_ID,
             "repository_full_name": REPO_NAME, "admission_id": adm_id,
             "sequence": seq, "wa_id": wa_id, "candidate_commit": candidate,
             "candidate_tree": self.tree(candidate),
             "ivv_evidence_ref": ev[0], "ivv_evidence_sha256": ev[1],
             "previous_admission_tag_object": prev, "issuer": "dainius1234",
             "note": "test"}
        p.update(over)
        return self.tag(f"kai-admitted/{adm_id}", target, p, signer=signer)

    def change(self, parent, edits, msg="candidate"):
        """edits: {rel: text | None(delete)} applied on top of `parent`."""
        self.checkout(parent)
        for rel, txt in edits.items():
            if txt is None:
                self.git("rm", "-q", rel)
            else:
                self.write(rel, txt)
        return self.commit(msg)

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def run_checker(name, *args) -> tuple:
    pr = subprocess.run([sys.executable, str(CHECKERS / name)] + list(args),
                        capture_output=True, text=True, timeout=300)
    return pr.returncode, pr.stdout + pr.stderr


class Suite:
    """Runs every case declared in `cases`, derived from the caller's list,
    and prints one line per case. Exit 1 if any case fails or if the
    executed count differs from the declared count."""

    def __init__(self, title):
        self.title = title
        self.rows = []

    def expect(self, case_id, statement, rc, out, *, want_rc, must_contain):
        ok = (rc == want_rc) and all(s in out for s in must_contain)
        self.rows.append((case_id, ok))
        verdict = "PASS" if ok else "FAIL"
        print(f"  {case_id:<5} {verdict}  rc={rc} (want {want_rc})  {statement}")
        if not ok:
            missing = [s for s in must_contain if s not in out]
            print(f"        missing reason text: {missing}")
            print("        --- checker output ---")
            for line in out.strip().splitlines()[-20:]:
                print(f"        | {line}")
        return ok

    def finish(self, declared) -> int:
        executed = [c for c, _ in self.rows]
        failed = [c for c, ok in self.rows if not ok]
        print(f"\n  {self.title}: declared {len(declared)}, executed "
              f"{len(executed)}, passed {len(executed) - len(failed)}, "
              f"failed {len(failed)}")
        if sorted(set(executed)) != sorted(set(declared)):
            print(f"  DECLARED/EXECUTED MISMATCH: not executed "
                  f"{sorted(set(declared) - set(executed))}, undeclared "
                  f"{sorted(set(executed) - set(declared))}")
            return 1
        return 1 if failed else 0
