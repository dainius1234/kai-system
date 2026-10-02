#!/usr/bin/env python3
"""D379 v4.5 governed build driver (C1/C2 per Kai 2026-10-02). Runs INSIDE buildns.sh.

usage: driver.py <A|B> <E_sha256>
One combined transcript /d379/out/<L>.transcript: driver framing lines plus the raw
stdout+stderr bytes of each command as produced, in order. Created exclusively (no
overwrite). Steps stop at the first non-zero return code. Meta -> /d379/out/<L>.meta.json.
"""
import datetime, hashlib, json, os, subprocess, sys

L, E_SHA = sys.argv[1], sys.argv[2]
assert L in ("A", "B")
SELF = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
STEPS = [
    ("configure", ["/d379/src/configure", "--prefix=/opt/d379-py311", "--without-ensurepip"]),
    ("make", ["make"]),
    ("install", ["make", "install", "DESTDIR=/d379/stage"]),
]
utc = lambda: datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
meta = {"label": L, "driver_sha256": SELF, "E_sha256": E_SHA, "start_utc": utc(),
        "cwd": "/d379/build", "env_keys": sorted(os.environ), "return_codes": {}}
with open(f"/d379/out/{L}.transcript", "xb") as t:
    def frame(s):
        t.write(f"=== D379 {s}\n".encode()); t.flush()
    frame(f"BUILD {L} driver_sha256={SELF} E_sha256={E_SHA} start={meta['start_utc']} "
          f"SOURCE_DATE_EPOCH={os.environ.get('SOURCE_DATE_EPOCH')}")
    for name, argv in STEPS:
        frame(f"CMD {name} {json.dumps(argv)} cwd=/d379/build start={utc()}")
        p = subprocess.Popen(argv, cwd="/d379/build", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for chunk in iter(lambda: p.stdout.read(65536), b""):
            t.write(chunk)
        rc = p.wait()
        meta["return_codes"][name] = rc
        frame(f"RC {name} {rc} end={utc()}")
        if rc:
            break
    meta["end_utc"] = utc()
    frame(f"END {L} return_codes={json.dumps(meta['return_codes'], sort_keys=True)} end={meta['end_utc']}")
with open(f"/d379/out/{L}.meta.json", "x") as m:
    json.dump(meta, m, indent=1, sort_keys=True)
ok = len(meta["return_codes"]) == len(STEPS) and not any(meta["return_codes"].values())
sys.exit(0 if ok else 1)
