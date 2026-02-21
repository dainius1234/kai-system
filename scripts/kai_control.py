from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import messagebox, simpledialog

import urllib.request

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception:  # pragma: no cover
    AESGCM = None

try:
    import qrcode
except Exception:  # pragma: no cover
    qrcode = None

WORDLIST = [
    "alpha", "anchor", "arch", "ash", "atlas", "binary", "blade", "bloom", "brave", "cedar", "cipher", "cobalt",
    "comet", "crown", "delta", "ember", "falcon", "forge", "galaxy", "granite", "harbor", "helium", "ion", "jade",
    "keeper", "lantern", "matrix", "nebula", "onyx", "origin", "phoenix", "pulse", "quantum", "raven", "sable", "signal",
    "solstice", "summit", "titan", "umbra", "vector", "vertex", "violet", "warden", "zenith", "aurora", "beacon", "ciphered",
]

KEEPER_NAME = "Dainius"
PRIMARY_HANDLE = "0x81000001"
APP_DIR = Path.home() / ".kai-control"
APP_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = APP_DIR / "emergency_actions.log"
USB_TIMEOUT_SECONDS = int(os.getenv("KAI_USB_TIMEOUT", "15"))


def _run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _vault_headers() -> Dict[str, str]:
    token = os.getenv("VAULT_ROOT_TOKEN") or os.getenv("VAULT_TOKEN")
    if not token:
        raise RuntimeError("VAULT_ROOT_TOKEN/VAULT_TOKEN required")
    return {"X-Vault-Token": token, "Content-Type": "application/json"}


def _vault_addr() -> str:
    return os.getenv("VAULT_ADDR", "http://127.0.0.1:8200").rstrip("/")


def vault_write(path: str, value: str) -> None:
    addr = _vault_addr()
    payload = json.dumps({"data": {"value": value}}).encode("utf-8")
    req = urllib.request.Request(f"{addr}/v1/{path}", data=payload, headers=_vault_headers(), method="POST")
    urllib.request.urlopen(req, timeout=5).read()


def vault_read(path: str) -> Optional[str]:
    addr = _vault_addr()
    req = urllib.request.Request(f"{addr}/v1/{path}", headers=_vault_headers(), method="GET")
    try:
        data = json.loads(urllib.request.urlopen(req, timeout=5).read().decode("utf-8"))
    except Exception:
        return None
    if isinstance(data.get("data"), dict):
        if "value" in data["data"]:
            return str(data["data"]["value"])
        inner = data["data"].get("data", {})
        if "value" in inner:
            return str(inner["value"])
    return None


def list_usb_mounts() -> List[Path]:
    mounts: List[Path] = []
    for root in [Path("/media") / os.getenv("USER", ""), Path("/run/media") / os.getenv("USER", "")]:
        if root.exists():
            mounts.extend([p for p in root.iterdir() if p.is_dir()])
    return mounts


def wait_for_usb_mounts(timeout_s: int = USB_TIMEOUT_SECONDS) -> List[Path]:
    deadline = time.time() + max(timeout_s, 1)
    mounts: List[Path] = []
    while time.time() < deadline:
        mounts = list_usb_mounts()
        if mounts:
            return mounts
        time.sleep(0.5)
    # adaptive extension for slow media
    slow_deadline = time.time() + 5
    while time.time() < slow_deadline:
        mounts = list_usb_mounts()
        if mounts:
            return mounts
        time.sleep(0.5)
    return mounts


@dataclass
class UsbKey:
    mount: Path
    pub: Path
    priv: Path


class KeeperRecoveryManager:
    def __init__(self) -> None:
        self.test_mode = os.getenv("KAI_CONTROL_TEST_MODE", "false").lower() == "true"

    def tpm_handle_verified(self) -> bool:
        if self.test_mode:
            return True
        if shutil.which("tpm2_readpublic") is None:
            return False
        proc = _run(["tpm2_readpublic", "-c", PRIMARY_HANDLE], check=False)
        return proc.returncode == 0

    def discover_keys(self) -> List[UsbKey]:
        keys: List[UsbKey] = []
        for m in list_usb_mounts():
            pub = m / "kai-primary.pub"
            priv = m / "kai-primary.priv"
            if pub.exists() and priv.exists():
                keys.append(UsbKey(mount=m, pub=pub, priv=priv))
        return keys

    def sealed_primary_exists(self) -> bool:
        return vault_read("secret/kai/keeper_primary_pubhash") is not None

    def sealed_backup_exists(self) -> bool:
        return vault_read("secret/kai/keeper_backup_pubhash") is not None

    def seal_primary(self, usb_mount: Path) -> str:
        pub = usb_mount / "kai-primary.pub"
        priv = usb_mount / "kai-primary.priv"
        if self.test_mode:
            seed = secrets.token_bytes(64)
            pub.write_bytes(hashlib.sha256(seed).digest())
            priv.write_bytes(seed)
        else:
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                seed = td_path / "seed.bin"
                primary = td_path / "primary.ctx"
                loaded = td_path / "loaded.ctx"
                seed.write_bytes(secrets.token_bytes(32))
                _run(["tpm2_createprimary", "-C", "o", "-c", str(primary)])
                _run(["tpm2_create", "-C", str(primary), "-u", str(pub), "-r", str(priv), "-i", str(seed)])
                _run(["tpm2_load", "-C", str(primary), "-u", str(pub), "-r", str(priv), "-c", str(loaded)])
                _run(["tpm2_evictcontrol", "-C", "o", "-c", str(loaded), PRIMARY_HANDLE])
        pubhash = _sha256_file(pub)
        vault_write("secret/kai/keeper_primary_pubhash", pubhash)
        return pubhash

    def seal_backup(self, backup_mount: Path, primary_mount: Path) -> str:
        src_pub = primary_mount / "kai-primary.pub"
        src_priv = primary_mount / "kai-primary.priv"
        dst_pub = backup_mount / "kai-primary.pub"
        dst_priv = backup_mount / "kai-primary.priv"
        shutil.copy2(src_pub, dst_pub)
        shutil.copy2(src_priv, dst_priv)
        pubhash = _sha256_file(dst_pub)
        vault_write("secret/kai/keeper_backup_pubhash", pubhash)
        return pubhash

    def key_presence(self) -> Dict[str, bool]:
        expected_primary = vault_read("secret/kai/keeper_primary_pubhash")
        expected_backup = vault_read("secret/kai/keeper_backup_pubhash")
        found_primary = False
        found_backup = False
        for key in self.discover_keys():
            pubhash = _sha256_file(key.pub)
            if expected_primary and pubhash == expected_primary:
                found_primary = True
            if expected_backup and pubhash == expected_backup:
                found_backup = True
        return {"primary": found_primary, "backup": found_backup}

    def validate_inserted_key(self) -> Optional[str]:
        presence = self.key_presence()
        if presence["primary"] and presence["backup"]:
            return "both"
        if presence["primary"]:
            return "usb"
        if presence["backup"]:
            return "backup"
        return None

    def can_destructive_actions(self) -> Tuple[bool, str]:
        presence = self.key_presence()
        if not (presence["primary"] and presence["backup"]):
            return False, "Insert both keys for this action"
        if not self.tpm_handle_verified():
            return False, "TPM handle verification failed"
        return True, "both"

    def generate_paper_recovery(self) -> Dict[str, str]:
        if AESGCM is None:
            raise RuntimeError("cryptography package is required")
        seed = secrets.token_bytes(32)
        words = [WORDLIST[b % len(WORDLIST)] for b in seed[:24]]
        phrase = " ".join(words)
        key = hashlib.sha256(phrase.encode("utf-8")).digest()
        aes = AESGCM(key)
        nonce = secrets.token_bytes(12)
        encrypted = aes.encrypt(nonce, seed, b"kai-recovery")
        payload = base64.b64encode(nonce + encrypted).decode("ascii")
        vault_write("secret/kai/keeper_recovery_hint", hashlib.sha256(payload.encode("utf-8")).hexdigest())
        if qrcode is not None:
            qr = qrcode.make(payload)
            qr.save(APP_DIR / "recovery_qr.png")
        (APP_DIR / "recovery_words.txt").write_text(phrase + "\n", encoding="utf-8")
        return {"payload": payload, "words": phrase, "qr_path": str(APP_DIR / "recovery_qr.png")}

    def restore_from_paper(self, payload: str, words: str, mount: Path) -> None:
        if AESGCM is None:
            raise RuntimeError("cryptography package is required")
        phrase = " ".join(words.strip().split())
        key = hashlib.sha256(phrase.encode("utf-8")).digest()
        raw = base64.b64decode(payload.encode("ascii"))
        nonce, enc = raw[:12], raw[12:]
        seed = AESGCM(key).decrypt(nonce, enc, b"kai-recovery")
        pub = mount / "kai-primary.pub"
        priv = mount / "kai-primary.priv"
        pub.write_bytes(hashlib.sha256(seed).digest())
        priv.write_bytes(seed)
        pubhash = _sha256_file(pub)
        vault_write("secret/kai/keeper_primary_pubhash", pubhash)


def log_action(method: str, action: str, signature: str) -> None:
    line = f"emergency action by keeper={KEEPER_NAME}, method={method}, action={action}, ts={_now()}, sig={signature}\n"
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


def kill_executor(method: str) -> None:
    _run(["docker", "compose", "-f", "docker-compose.sovereign.yml", "stop", "executor"], check=False)
    log_action(method, "kill", "TPM")


def rollback_memu(method: str) -> None:
    url = os.getenv("MEMU_URL", "http://localhost:8001")
    with urllib.request.urlopen(f"{url}/memory/stats", timeout=3) as r:
        stats = json.loads(r.read().decode("utf-8"))
    commits = stats.get("commits") or []
    if not commits:
        raise RuntimeError("No commits available for rollback")
    version = commits[0]["commit_id"]
    req = urllib.request.Request(f"{url}/revert?version={version}", data=b"", method="POST")
    urllib.request.urlopen(req, timeout=5).read()
    log_action(method, f"rollback:{version}", "TPM")


def unlock_logs(method: str) -> str:
    try:
        output = _run(["docker", "logs", "--tail", "50", "executor"], check=False).stdout
    except Exception:
        output = "logs unavailable"
    log_action(method, "unlock_logs", "TPM")
    return output


class KaiControlUI:
    def __init__(self) -> None:
        self.mgr = KeeperRecoveryManager()
        self.method: Optional[str] = None
        self.root = tk.Tk()
        self.root.title("kai-control")
        self.status = tk.Label(self.root, text="Insert USB", width=80)
        self.status.pack(pady=8)

        self.btn_seal = tk.Button(self.root, text="Seal primary", command=self._seal_primary)
        self.btn_backup = tk.Button(self.root, text="Seal backup", command=self._seal_backup)
        self.btn_paper = tk.Button(self.root, text="Generate paper recovery", command=self._paper)
        self.btn_restore = tk.Button(self.root, text="Restore from paper", command=self._restore)

        self.btn_kill = tk.Button(self.root, text="Kill", command=self._kill)
        self.btn_roll = tk.Button(self.root, text="Rollback", command=self._rollback)
        self.btn_logs = tk.Button(self.root, text="Unlock Logs", command=self._logs)

        for b in [self.btn_seal, self.btn_backup, self.btn_paper, self.btn_restore, self.btn_kill, self.btn_roll, self.btn_logs]:
            b.pack_forget()

        self._refresh()

    def _first_usb(self) -> Path:
        mounts = list_usb_mounts()
        if not mounts:
            raise RuntimeError("No USB detected")
        return mounts[0]

    def _seal_primary(self) -> None:
        mount = self._first_usb()
        h = self.mgr.seal_primary(mount)
        messagebox.showinfo("kai-control", f"Primary sealed: {h[:16]}...")

    def _seal_backup(self) -> None:
        mounts = list_usb_mounts()
        if len(mounts) < 2:
            raise RuntimeError("Insert second USB for backup")
        h = self.mgr.seal_backup(mounts[1], mounts[0])
        messagebox.showinfo("kai-control", f"Backup sealed: {h[:16]}...")

    def _paper(self) -> None:
        out = self.mgr.generate_paper_recovery()
        messagebox.showinfo("kai-control", f"24 words:\n{out['words']}\n\nQR: {out['qr_path']}")

    def _restore(self) -> None:
        payload = simpledialog.askstring("kai-control", "Paste paper QR payload")
        words = simpledialog.askstring("kai-control", "Type 24 words")
        if not payload or not words:
            return
        self.mgr.restore_from_paper(payload, words, self._first_usb())
        messagebox.showinfo("kai-control", "Restored and resealed on USB")

    def _kill(self) -> None:
        allowed, reason = self.mgr.can_destructive_actions()
        if not allowed:
            messagebox.showwarning("kai-control", reason)
            return
        kill_executor("both")
        messagebox.showinfo("kai-control", "Executor stopped")

    def _rollback(self) -> None:
        allowed, reason = self.mgr.can_destructive_actions()
        if not allowed:
            messagebox.showwarning("kai-control", reason)
            return
        rollback_memu("both")
        messagebox.showinfo("kai-control", "Rollback sent")

    def _logs(self) -> None:
        logs = unlock_logs(self.method or "usb")
        messagebox.showinfo("kai-control", logs[:3500])

    def _refresh(self) -> None:
        for b in [self.btn_seal, self.btn_backup, self.btn_paper, self.btn_restore, self.btn_kill, self.btn_roll, self.btn_logs]:
            b.pack_forget()

        method = self.mgr.validate_inserted_key()
        primary = self.mgr.sealed_primary_exists()
        backup = self.mgr.sealed_backup_exists()

        if not primary:
            self.status.config(text="Insert primary USB and click Seal")
            self.btn_seal.pack(pady=4)
        elif primary and not backup:
            self.status.config(text="Primary OK. Insert second USB to seal backup.")
            self.btn_backup.pack(pady=4)
            self.btn_paper.pack(pady=4)
            self.btn_restore.pack(pady=4)
        elif method:
            self.method = method
            allowed, _ = self.mgr.can_destructive_actions()
            if allowed:
                self.status.config(text="Both keys recognized. Destructive controls unlocked.")
                self.btn_kill.pack(pady=4)
                self.btn_roll.pack(pady=4)
            else:
                self.status.config(text="Single key recognized. Insert both keys for Kill/Rollback.")
            self.btn_logs.pack(pady=4)
            self.btn_paper.pack(pady=4)
            self.btn_restore.pack(pady=4)
        else:
            self.status.config(text="Key not recognized. Try another USB or paper restore.")
            self.btn_restore.pack(pady=4)

        self.root.after(2000, self._refresh)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    KaiControlUI().run()
