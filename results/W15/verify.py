#!/usr/bin/env python3
"""W15 replay verification against production (API requests only; no restart).

    verify.py OUT.json [DEPTHS]     DEPTHS default 8192,32768,65536

For each depth, a prompt of exactly DEPTH token ids that this server has never seen (a fixed label + the depth, no cache
entry can exist: the log must say cached 0 / cache_src none on its first send):
  cold      the prompt, greedy, 64 tokens, ignore_eos: computed from scratch (cached 0)
  replay    the identical ids again: must resume at n - 64 and give the byte-identical 64 tokens
  replay2   once more (same)
negative controls (each sent once, after the base; the cache may only use a strict prefix the variant shares):
  start     one token changed at position 3 (right after the special tokens): expect cached 0
  middle    one token changed at 0.52 n: expect cached <= the change position (the last session mark before it,
            GLM53_TF_SESSION_EVERY = 16,384), the rest re-prefilled
  end10     one token changed at n - 10: the n - 64 snapshot lies before the change: expect cached n - 64
  end100    one token changed at n - 100: the n - 64 snapshot contains the change: expect the mark before it
  other     a different prompt of the same length: expect cached 0
Every row: client TTFT, the request log's cached / cache_src / pieces / prefill_s / first_s, the 64-token output sha.
"""
import json, sys, zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay as rp  # noqa: E402  (stream(), post_json(), FILLER)

MODEL = rp.MODEL
WORDS = ("amber basalt cedar delta ember fjord granite harbor iris juniper kelp lagoon meadow nectar obsidian prairie "
         "quartz river sierra tundra umber valley willow xenon yarrow zephyr").split()


def ids_of(label, depth):
    lines, i = [f"[{label}]"], 0
    while len(lines) < depth // 8 + 64:
        lines.append(f"{label}-{i:05d}: " + " ".join(WORDS[zlib.crc32(f"{label}/{i}/{j}".encode()) % len(WORDS)] for j in range(5)) + ".")
        i += 1
    ids = rp.post_json("/tokenize", {"model": MODEL, "prompt": "\n".join(lines)})["tokens"]
    assert len(ids) >= depth
    return ids[:depth]


def other_token(ids, k):
    """A token id that differs from ids[k], taken from elsewhere in the prompt (a normal text token)."""
    for j in range(k + 1, len(ids)):
        if ids[j] != ids[k] and ids[j] > 1000:
            return ids[j]
    raise ValueError


def send(ids):
    r = rp.stream("/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": 64, "temperature": 0, "top_p": 1,
                                      "ignore_eos": True})
    return r


def main():
    out = sys.argv[1]
    depths = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "8192,32768,65536").split(",")]
    res = []
    for n in depths:
        base = ids_of(f"w15-verify-{n}", n)
        rows = {}
        for name in ("cold", "replay", "replay2"):
            rows[name] = send(base)
        variants = {}
        for name, k in (("start", 3), ("middle", int(n * 0.52)), ("end10", n - 10), ("end100", n - 100)):
            v = list(base); v[k] = other_token(base, k); variants[name] = (v, k)
        variants["other"] = (ids_of(f"w15-verify-other-{n}", n), None)
        for name, (v, k) in variants.items():
            rows[name] = send(v); rows[name]["changed_at"] = k
        c = rows["cold"]
        summ = dict(n=n, cold_cached=c["log"]["cached"], cold_src=c["log"]["cache_src"],
                    replay_identical=rows["replay"]["sha"] == c["sha"] and rows["replay2"]["sha"] == c["sha"],
                    rows={k: dict(ttft=r["ttft"], tps=round(n / r["ttft"], 1) if r["ttft"] else None, sha=r["sha"],
                                  changed_at=r.get("changed_at"), **{x: r["log"][x] for x in ("cached", "cache_src", "pieces",
                                                                                                "prefill_s", "first_s", "n")})
                          for k, r in rows.items()})
        res.append(summ)
        print(f"== depth {n}: cold cached {summ['cold_cached']} ({summ['cold_src']}); replays identical to cold: "
              f"{summ['replay_identical']}", flush=True)
        for k, r in summ["rows"].items():
            print(f"   {k:8s} changed_at {str(r['changed_at']):>6}  cached {r['cached']:>6} {r['cache_src']:5s} pieces "
                  f"{r['pieces']:>2}  prefill_s {r['prefill_s']:>8}  first_s {r['first_s']:>7}  client ttft {r['ttft']:>7} s "
                  f"({r['tps']:>10} tok/s)  sha {r['sha']}", flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
