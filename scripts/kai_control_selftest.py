from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("kai_control", ROOT / "scripts" / "kai_control.py")
kc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = kc
spec.loader.exec_module(kc)


class _MemVault:
    data = {}


def _write(path: str, value: str) -> None:
    _MemVault.data[path] = value


def _read(path: str):
    return _MemVault.data.get(path)


def main() -> None:
    kc.vault_write = _write
    kc.vault_read = _read
    os.environ["KAI_CONTROL_TEST_MODE"] = "true"

    with tempfile.TemporaryDirectory() as td:
        usb1 = Path(td) / "usb1"
        usb2 = Path(td) / "usb2"
        usb3 = Path(td) / "usb3"
        usb1.mkdir(); usb2.mkdir(); usb3.mkdir()

        mgr = kc.KeeperRecoveryManager()
        h1 = mgr.seal_primary(usb1)
        assert h1 == kc._sha256_file(usb1 / "kai-primary.pub")
        h2 = mgr.seal_backup(usb2, usb1)
        assert h1 == h2

        rec = mgr.generate_paper_recovery()
        mgr.restore_from_paper(rec["payload"], rec["words"], usb3)
        assert (usb3 / "kai-primary.pub").exists()

    print("kai-control selftest passed")


if __name__ == "__main__":
    main()
