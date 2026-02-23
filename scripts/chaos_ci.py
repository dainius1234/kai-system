
"""
chaos_ci.py: Chaos testing harness for Kai system microservices.

Features:
- CLI interface for scenario selection, dry-run, and verbosity
- Robust error handling and logging
- Simulates service failures and validates SLOs via game-day scorecard
"""

import os
import random
import signal
import subprocess
import time
import argparse
import logging
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



def start(verbose: bool = False) -> List[subprocess.Popen]:
    procs = []
    for i, (cmd, env_extra) in enumerate(zip(SERVICES, PORT_ENVS)):
        env = os.environ.copy()
        env.update(env_extra)
        stdout = None if verbose else subprocess.DEVNULL
        stderr = None if verbose else subprocess.DEVNULL
        try:
            p = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)
            procs.append(p)
            logging.info(f"Started service {i}: {' '.join(cmd)}")
        except Exception as e:
            logging.error(f"Failed to start service {i}: {e}")
            raise
    return procs



def restart(procs: List[subprocess.Popen], idx: int, verbose: bool = False) -> None:
    env = os.environ.copy()
    env.update(PORT_ENVS[idx])
    stdout = None if verbose else subprocess.DEVNULL
    stderr = None if verbose else subprocess.DEVNULL
    try:
        procs[idx] = subprocess.Popen(SERVICES[idx], env=env, stdout=stdout, stderr=stderr)
        logging.info(f"Restarted service {idx}: {' '.join(SERVICES[idx])}")
    except Exception as e:
        logging.error(f"Failed to restart service {idx}: {e}")
        raise



def stop_all(procs: List[subprocess.Popen]) -> None:
    for i, p in enumerate(procs):
        try:
            p.terminate()
            logging.info(f"Terminated service {i}")
        except Exception as e:
            logging.warning(f"Failed to terminate service {i}: {e}")
    for i, p in enumerate(procs):
        try:
            p.wait(timeout=3)
        except Exception:
            try:
                p.kill()
                logging.info(f"Killed service {i}")
            except Exception as e:
                logging.warning(f"Failed to kill service {i}: {e}")



def run_scenario(procs: List[subprocess.Popen], scenario: str, verbose: bool = False) -> None:
    if scenario == "random_kill":
        kill_idx = random.randrange(len(procs))
        procs[kill_idx].send_signal(signal.SIGKILL)
        logging.info(f"Sent SIGKILL to service {kill_idx}")
        time.sleep(1)
        restart(procs, kill_idx, verbose=verbose)
    elif scenario == "memu_degrade":
        memu_idx = 1
        procs[memu_idx].send_signal(signal.SIGTERM)
        logging.info(f"Sent SIGTERM to memu-core (service {memu_idx})")
        time.sleep(1)
        restart(procs, memu_idx, verbose=verbose)
        time.sleep(2)
    else:
        logging.warning(f"Unknown scenario: {scenario}")


def main():
    parser = argparse.ArgumentParser(description="Kai Chaos CI: Simulate failures and validate SLOs.")
    parser.add_argument("--dry-run", action="store_true", help="Only print actions, do not execute")
    parser.add_argument("--scenario", choices=["random_kill", "memu_degrade", "all"], default="all", help="Chaos scenario to run")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="[%(levelname)s] %(message)s")
    random.seed(42)

    if args.dry_run:
        print("[DRY RUN] Would start services, run scenario(s), and validate SLOs.")
        return

    procs = start(verbose=args.verbose)
    try:
        time.sleep(2)
        if args.scenario in ("random_kill", "all"):
            run_scenario(procs, "random_kill", verbose=args.verbose)
        if args.scenario in ("memu_degrade", "all"):
            run_scenario(procs, "memu_degrade", verbose=args.verbose)

        rc = subprocess.run(["make", "game-day-scorecard"], check=False).returncode
        if rc != 0:
            logging.error("Game-day scorecard failed SLOs under chaos pressure.")
            raise SystemExit(rc)
        print("chaos ci passed")
    except Exception as e:
        logging.error(f"Exception during chaos run: {e}")
        raise
    finally:
        stop_all(procs)



if __name__ == "__main__":
    main()
