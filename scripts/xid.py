#!/usr/bin/env python3
"""Classify NVIDIA Xid events from kernel log text (``journalctl -k`` / ``dmesg``) on stdin.

``NVRM: Xid (PCI:0000:01:00): 79, pid=..., GPU has fallen off the bus.`` -> one record a line. The classes follow
NVIDIA's Xid catalog: ``fatal`` means the GPU (or its context) needs a reset/reboot and a serving process on it is
dead or about to hang; ``app`` means a CUDA program faulted (our server, if it was the one running); ``info`` is
benign bookkeeping. Used by ``scripts/serve.sh xid`` and ``watch``.

    journalctl -k --since '-1h' | scripts/xid.py [--fail-on fatal|app|any] [--json]

Exit 0 when nothing at or above ``--fail-on`` was seen (default ``fatal``), 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# Xid -> (class, short meaning). Anything not listed is "app" (unknown faults are treated as worth a look).
XIDS: dict[int, tuple[str, str]] = {
    8: ("app", "GPU stopped processing (timeout)"),
    13: ("app", "graphics engine exception"),
    31: ("app", "GPU memory page fault (MMU)"),
    32: ("app", "invalid or corrupted push buffer stream"),
    38: ("app", "driver firmware error"),
    43: ("info", "GPU stopped processing (a context was reset)"),
    45: ("info", "preemptive cleanup after a process exit"),
    48: ("fatal", "double-bit ECC error"),
    56: ("app", "display engine error"),
    61: ("fatal", "internal micro-controller breakpoint/warning"),
    62: ("fatal", "internal micro-controller halt"),
    63: ("info", "ECC page retirement / row remap recorded"),
    64: ("fatal", "ECC page retirement / row remap failure"),
    68: ("app", "video processor exception"),
    69: ("app", "graphics engine class error"),
    74: ("fatal", "NVLink error"),
    79: ("fatal", "GPU has fallen off the bus"),
    92: ("info", "high single-bit ECC error rate"),
    94: ("app", "contained ECC error (the affected process was killed)"),
    95: ("fatal", "uncontained ECC error (reset the GPU)"),
    109: ("app", "context switch timeout"),
    119: ("fatal", "GSP RPC timeout"),
    120: ("fatal", "GSP error"),
    121: ("info", "C2C link corrected error"),
    136: ("fatal", "link training failure"),
    140: ("fatal", "unrecovered ECC error"),
    143: ("fatal", "GSP firmware error"),
    154: ("fatal", "GPU recovery action required (see the message for reset/reboot)"),
}
RANK = {"info": 0, "app": 1, "fatal": 2}
LINE = re.compile(r"NVRM: Xid \((?P<dev>[^)]*)\):\s*(?P<xid>\d+),?\s*(?P<rest>.*)")


def parse(text: str) -> list[dict[str, object]]:
    out = []
    for line in text.splitlines():
        m = LINE.search(line)
        if not m:
            continue
        xid = int(m["xid"])
        cls, meaning = XIDS.get(xid, ("app", "unlisted Xid"))
        pid = re.search(r"pid=(\d+)", m["rest"])
        name = re.search(r"name=([^,]+)", m["rest"])
        out.append({"xid": xid, "class": cls, "meaning": meaning, "device": m["dev"],
                    "pid": int(pid[1]) if pid else None, "process": name[1].strip() if name else None,
                    "line": line.strip()})
    return out


def worst(events: list[dict[str, object]]) -> str | None:
    return max((str(e["class"]) for e in events), key=RANK.__getitem__, default=None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--fail-on", choices=["fatal", "app", "any"], default="fatal")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--label", default="", help="prefix for the printed lines (e.g. the node)")
    args = ap.parse_args(argv)
    events = parse(sys.stdin.read())
    if args.json:
        print(json.dumps(events, indent=1))
    else:
        for e in events:
            print(f"{args.label}Xid {e['xid']} [{e['class']}] {e['meaning']}"
                  + (f" (pid {e['pid']} {e['process'] or ''})" if e["pid"] else ""))
    floor = {"any": 0, "app": 1, "fatal": 2}[args.fail_on]
    return 1 if any(RANK[str(e["class"])] >= floor for e in events) else 0


if __name__ == "__main__":
    sys.exit(main())
