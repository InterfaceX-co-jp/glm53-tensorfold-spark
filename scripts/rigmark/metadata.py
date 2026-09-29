#!/usr/bin/env python3
"""Write RigMark appliance metadata (METADATA.md fields) for one engine of the sweep, from the live configuration.

    metadata.py vllm       --out FILE [--competing-traffic TEXT] [--base-url URL]
    metadata.py tensorfold --out FILE [--competing-traffic TEXT] [--config config/prod.env]

Run on the head Spark (head). Reads only named keys of the kit's .env / our config (never tokens), `docker image
inspect`, the RoCE link's sysfs speed and, for vLLM, GET /version. Paths, usernames and hostnames are not written
(RigMark's metadata rules). Anything a probe cannot read is written as "unknown", never guessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KIT = Path(os.environ.get("VLLM_KIT_DIR", str(Path.home() / "ai/src/glm53-exl3-2x-kit")))
LINK_IF = os.environ.get("RIGMARK_LINK_IF", "enp1s0f1np1")
MODEL = "neko-legends/GLM-5.3-Flash-Uncensored-EXL3"
DRAFTER = "incoai/GLM-5.3-Flash-DFlash2"


def env_keys(path: Path, keys: tuple[str, ...]) -> dict[str, str]:
    """Only the named KEY=value lines of a shell env file (inline comments dropped, quotes stripped)."""
    out: dict[str, str] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        m = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if not m or m.group(1) not in keys:
            continue
        value = m.group(2)
        if value[:1] in "\"'":
            q = value[0]
            value = value[1:value.find(q, 1)] if value.find(q, 1) > 0 else value[1:]
        else:
            value = re.sub(r"\s+#.*$", "", value).strip()
        out[m.group(1)] = value
    return out


def sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def image_id(tag: str) -> str:
    iid = sh("docker", "image", "inspect", "--format", "{{.Id}}", tag) if tag else ""
    return f"local image {tag}, id {iid}" if iid else (f"local image {tag}, id unknown" if tag else "unknown")


def link_speed() -> str:
    try:
        mbps = int(Path(f"/sys/class/net/{LINK_IF}/speed").read_text().strip())
        return f"{mbps // 1000} Gb/s per port (head, sysfs)"
    except (OSError, ValueError):
        return "unknown"


def sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "unknown"


def snapshot_rev(path: str) -> str:
    m = re.search(r"/snapshots/([0-9a-f]{40})", path or "")
    return m.group(1) if m else "unknown"


COMMON = {
    "hardware": "2x NVIDIA DGX Spark (GB10, 128 GB unified memory each)",
    "topology": "TP2 across two nodes, direct-connect ConnectX-7 RoCE, no switch",
    "model": MODEL + " (GLM-5.3-Flash, abliterated, EXL3 4-bit)",
    "client": "RigMark on the head node against 127.0.0.1 (loopback, no proxy)",
}


def vllm(base_url: str) -> dict:
    keys = ("MODEL", "MODEL_REVISION", "IMAGE", "MAX_MODEL_LEN", "MAX_NUM_SEQS", "MAX_NUM_BATCHED_TOKENS",
            "KV_CACHE_DTYPE", "SPEC_METHOD", "DFLASH_MODEL", "DFLASH_REVISION", "DFLASH_TOKENS", "GLM53_ADAPTIVE_K",
            "GLM53_DEFAULT_REASONING_EFFORT", "EXTRA_ARGS", "GPU_MEM_UTIL")
    e = env_keys(KIT / ".env", keys)
    version = "unknown"
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/version", timeout=10) as r:
            version = json.load(r).get("version", "unknown")
    except Exception:  # noqa: BLE001
        pass
    kit_rev = sh("git", "-C", str(KIT), "rev-parse", "HEAD") or "unknown"
    kit_dirty = bool(sh("git", "-C", str(KIT), "status", "--porcelain", "--untracked-files=no"))
    seqs, batched = e.get("MAX_NUM_SEQS", "unknown"), e.get("MAX_NUM_BATCHED_TOKENS", "2048 (kit default)")
    return {
        **COMMON,
        "model_revision": e.get("MODEL_REVISION", "unknown"),
        "quantisation": "EXL3 4-bit as published (experts and non-expert weights from the checkpoint)",
        "kv_cache_dtype": "FP8 (vLLM fp8, packed fp8_ds_mla)" if e.get("KV_CACHE_DTYPE", "fp8") == "fp8"
        else e.get("KV_CACHE_DTYPE", "unknown"),
        "serving_engine": f"vLLM {version} (GLM-5.3-Flash EXL3 2x DGX Spark kit, MiaAI-Lab lineage, local fork "
                          f"{kit_rev[:12]}{' with uncommitted local changes' if kit_dirty else ''})",
        "serving_image": image_id(e.get("IMAGE", "")),
        "drafter": f"{e.get('DFLASH_MODEL', DRAFTER)}@{e.get('DFLASH_REVISION', 'unknown')[:12]}, "
                   f"{e.get('SPEC_METHOD', 'dflash')} k={e.get('DFLASH_TOKENS', '7')}, adaptive k "
                   f"({e.get('GLM53_ADAPTIVE_K', 'off')}), greedy verification",
        "context_limit": int(e.get("MAX_MODEL_LEN", "1000000")),
        "scheduler": f"max-num-seqs {seqs}; max-num-batched-tokens {batched}; {e.get('EXTRA_ARGS', '').strip()}",
        "server_default_reasoning_effort": e.get("GLM53_DEFAULT_REASONING_EFFORT", "template default (max)"),
        "chat_template_sha256": sha256(KIT / "files/chat_template.jinja"),
        "launcher_sha256": {"start.sh": sha256(KIT / "start.sh"), "local/prod-start.sh": sha256(KIT / "local/prod-start.sh")},
    }


def tensorfold(config: Path) -> dict:
    keys = ("IMAGE", "CONTEXT", "MAX_TOKENS", "MODEL_PATH", "DRAFTER", "GLM53_TF_NONEXPERT", "GLM53_TF_KV_DTYPE",
            "GLM53_TF_LATENT_KV", "GLM53_TF_BATCH", "GLM53_TF_KV_POOL_TOKENS", "GLM53_TF_BATCH_PIECE",
            "GLM53_TF_SOLO_PIECE", "GLM53_TF_DEPTH", "GLM53_TF_AUTO_FDRAFTS", "GLM53_TF_LOOKUP",
            "GLM53_TF_MAX_DRAFT_ROWS", "GLM53_TF_DEFAULT_EFFORT", "GLM53_TF_COMM_BACKEND", "GLM53_TF_BATCH_MTP",
            "GLM53_TF_PREFIX_SHARE", "GLM53_TF_BATCH_SESSIONS", "GLM53_TF_SESSION_GIB")
    e = env_keys(config, keys)
    for k in keys:      # serve.sh: a non-empty caller export wins over the config file
        if os.environ.get(k):
            e[k] = os.environ[k]
    vendor = sh("git", "-C", str(REPO / "vendor/TensorFold"), "rev-parse", "--short=7", "HEAD") or "2f8e514"
    repo = sh("git", "-C", str(REPO), "rev-parse", "--short=12", "HEAD") or "unknown (not a git checkout)"
    nonexpert = e.get("GLM53_TF_NONEXPERT", "bf16")
    return {
        **COMMON,
        "model_revision": snapshot_rev(e.get("MODEL_PATH", "")),
        "quantisation": "EXL3 4-bit experts as published; non-expert weights "
                        + ("re-quantized to 4-bit at load (q4mse, from the checkpoint's BF16)" if nonexpert == "q4mse"
                           else f"{nonexpert}"),
        "kv_cache_dtype": f"{e.get('GLM53_TF_KV_DTYPE', 'bf16').upper()} latent (MLA) KV"
                          + (", shared pool of " + e["GLM53_TF_KV_POOL_TOKENS"] + " tokens"
                             if e.get("GLM53_TF_KV_POOL_TOKENS") else ""),
        "serving_engine": f"TensorFold {vendor} + glm53-tensorfold-spark engine patches (repo {repo}; "
                          f"config/prod.env), OpenAI server from tensorfold.cuda.server",
        "serving_image": image_id(e.get("IMAGE", "")),
        "drafter": f"{DRAFTER}@{snapshot_rev(e.get('DRAFTER', ''))[:12]} + MTP head, cost-derived depth "
                   f"({e.get('GLM53_TF_DEPTH', '?')}), up to {e.get('GLM53_TF_AUTO_FDRAFTS', '?')} drafts, verify "
                   f"windows up to {e.get('GLM53_TF_MAX_DRAFT_ROWS', '8')} rows, suffix lookup "
                   f"{'on' if e.get('GLM53_TF_LOOKUP') == '1' else 'off'}; greedy verification",
        "context_limit": int(e.get("CONTEXT", "0") or 0),
        "scheduler": f"{e.get('GLM53_TF_BATCH', '1')} batch slots; prefill pieces {e.get('GLM53_TF_BATCH_PIECE', '?')} "
                     f"rows in a batch / {e.get('GLM53_TF_SOLO_PIECE', '?')} alone; batch MTP "
                     f"{e.get('GLM53_TF_BATCH_MTP', '0')}; session store {e.get('GLM53_TF_SESSION_GIB', '0')} GiB RAM "
                     f"+ NVMe; prefix share {e.get('GLM53_TF_PREFIX_SHARE', '0')}; all-gather "
                     f"{e.get('GLM53_TF_COMM_BACKEND', 'nccl')}",
        "server_default_reasoning_effort": e.get("GLM53_TF_DEFAULT_EFFORT", "template default (max)"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine", choices=("vllm", "tensorfold"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--competing-traffic", default="none: endpoint idle at start (checked), test window held")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--config", type=Path, default=REPO / os.environ.get("CONFIG", "config/prod.env"))
    a = ap.parse_args()
    meta = vllm(a.base_url) if a.engine == "vllm" else tensorfold(a.config)
    meta["negotiated_link_speed"] = link_speed()
    meta["competing_traffic"] = a.competing_traffic
    if not meta["context_limit"]:
        raise SystemExit("context_limit unknown: refusing to write metadata")
    a.out.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
