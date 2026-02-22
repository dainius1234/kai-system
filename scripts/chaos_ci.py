from __future__ import annotations

import os
import random
import signal
import subprocess
import time
from typing import List

SERVICES = [
    ["python", "tool-gate/app.py"],
    ["python", "memu-core/app.py"],
    ["python", "langgraph/app.py"],
]
PORT_ENVS = [
    {"PORT": "19000", "TRUSTED_TOKENS_PATH": "security/trusted_tokens.txt", "MODE": "WORK", "REQUIRE_SIGNATURE": "true"},
    {"PORT": "19001"},
    {"PORT": "19007", "MEMU_URL": "http://127.0.0.1:19001", "TOOL_GATE_URL": "http://127.0.0.1:19000"},
]


def start() -> List[subprocess.Popen]:
    procs = []
    for cmd, env_extra in zip(SERVICES, PORT_ENVS):
        env = os.environ.copy()
        env.update(env_extra)
        procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    return procs


def stop_all(procs: List[subprocess.Popen]) -> None:
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    for p in procs:
        try:
            p.wait(timeout=3)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


def main() -> None:
    random.seed(42)
    procs = start()
    try:
        time.sleep(2)
        victim = random.choice(procs)
        victim.send_signal(signal.SIGTERM)
        time.sleep(1)
        # if victim died, restart it
        if victim.poll() is not None:
            idx = procs.index(victim)
            env = os.environ.copy()
            env.update(PORT_ENVS[idx])
            procs[idx] = subprocess.Popen(SERVICES[idx], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        rc = subprocess.run(["make", "game-day-scorecard"]).returncode
        if rc != 0:
            raise SystemExit(rc)
        print("chaos ci passed")
    finally:
        stop_all(procs)


if __name__ == "__main__":
    main()
