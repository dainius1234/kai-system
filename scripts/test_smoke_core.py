from __future__ import annotations

import subprocess


def main() -> int:
    # When core services are not running, the script should exit nonzero
    res = subprocess.run(["python3", "scripts/smoke_core.py"], capture_output=True, text=True)
    print(res.stdout)
    return res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
