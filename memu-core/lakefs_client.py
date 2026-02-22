from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class VersionCommit:
    commit_id: str
    branch: str
    timestamp: float
    message: str
    sha256: str


class LakeFSClient:
    """Lightweight local client with lakeFS-like branch/commit semantics."""

    def __init__(self) -> None:
        self._branches: Dict[str, Dict[str, Any]] = {"main": {"records": [], "state": {}}}
        self._commits: List[VersionCommit] = []

    def create_branch(self, source: str, branch: str) -> str:
        source_state = self._branches.get(source, {"records": [], "state": {}})
        self._branches[branch] = {
            "records": json.loads(json.dumps(source_state["records"])),
            "state": dict(source_state["state"]),
        }
        return branch

    def put_branch_state(self, branch: str, records: List[Dict[str, Any]], state: Dict[str, Any], message: str) -> VersionCommit:
        self._branches[branch] = {"records": json.loads(json.dumps(records)), "state": dict(state)}
        payload = json.dumps({"branch": branch, "records": records, "state": state, "message": message, "ts": time.time()}, sort_keys=True)
        sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        commit = VersionCommit(commit_id=sha[:12], branch=branch, timestamp=time.time(), message=message, sha256=sha)
        self._commits.append(commit)
        self._branches["main"] = self._branches[branch]
        return commit

    def revert(self, commit_id: str) -> None:
        index = next((i for i, c in enumerate(self._commits) if c.commit_id == commit_id), -1)
        if index < 0:
            raise KeyError(f"unknown commit: {commit_id}")
        records: List[Dict[str, Any]] = []
        state: Dict[str, Any] = {}
        for commit in self._commits[: index + 1]:
            data = self._branches.get(commit.branch)
            if data:
                records = json.loads(json.dumps(data["records"]))
                state = dict(data["state"])
        self._branches["main"] = {"records": records, "state": state}
        self._commits = self._commits[: index + 1]

    def latest_main(self) -> Dict[str, Any]:
        return self._branches["main"]

    def list_commits(self) -> List[VersionCommit]:
        return list(reversed(self._commits))
