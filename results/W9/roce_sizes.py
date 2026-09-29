"""W9: why W7's single-node loopback had bits_equal false at 1 MiB (tests/cuda/bench_roce.py loopback, 0350 harness).

Hypothesis A (harness race): loopback builds its inputs with ``roce.pattern`` (arange / mul / add / shift / cast
kernels) on the DEFAULT stream, then launches each rank's gather on its own torch side stream (created
non-blocking), with no wait: the gather's staging copy may read the input before the pattern kernels wrote it.
The larger the input, the longer the pattern kernels run, so the more likely the gather overtakes them.
Hypothesis B (transport): a size / chunk / slot boundary bug in the runtime (would also show with the sync).

For every size and mode, ``trials`` fresh inputs (new salt each), both ranks' first gather on side streams:
  race: exactly like loopback (no wait for the default stream)
  sync: torch.cuda.synchronize() between building the inputs and the gathers
then ``reuse``: the same inputs gathered again (inputs long written) -- a transport bug would still differ.
Reports per (size, mode) the number of trials that differed and, for the first bad trial, which output shard (own
input copy vs the peer's received slot) and the first / last differing byte.
"""

import dataclasses
import json
import sys

import torch

from tensorfold.families.glm5_next.cuda import roce


def main() -> None:
    sizes = [int(v) * 1024 for v in (sys.argv[1] if len(sys.argv) > 1 else "16,128,256,512,1024,2048,4096").split(",")]
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    torch.cuda.set_device(0)
    s = roce.settings()
    s = dataclasses.replace(s, max_bytes=max(sizes), timeout_s=10.0)
    hcas = roce.detect(s.hca_spec)[:s.hcas]
    print("HCAs:", [(h.name, h.gid_index, h.ipv4) for h in hcas], "slot", max(sizes), flush=True)
    rts = [roce.Runtime(rank=r, world=2, hcas=hcas, s=s) for r in (0, 1)]
    blobs = [rt.blob() for rt in rts]
    for rt in rts:
        rt.connect(blobs)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    salt = 0
    ok_all = True
    for n in sizes:
        for mode in (sys.argv[3].split(",") if len(sys.argv) > 3 else ("race", "sync")):
            bad, first = 0, None
            reuse_bad = 0
            for t in range(trials):
                salt += 1
                xs = [roce.pattern(r, n, salt) for r in (0, 1)]
                ys = [torch.full((2 * n,), 0xAB, dtype=torch.uint8, device="cuda") for _ in (0, 1)]
                if mode == "sync":
                    torch.cuda.synchronize()
                for r in (0, 1):
                    with torch.cuda.stream(streams[r]):
                        rts[r].gather(xs[r], ys[r])
                torch.cuda.synchronize()
                for rt in rts:
                    rt.check()
                exp = torch.cat([roce.pattern(r, n, salt) for r in (0, 1)])
                diff = [(ys[r] != exp).nonzero().flatten() for r in (0, 1)]
                if any(d.numel() for d in diff):
                    bad += 1
                    if first is None:
                        first = []
                        for r in (0, 1):
                            d = diff[r]
                            if d.numel():
                                own = d[(d // n) == r]
                                peer = d[(d // n) != r]
                                first.append({"rank": r, "bytes_diff": int(d.numel()),
                                              "own_shard_diff": int(own.numel()), "peer_shard_diff": int(peer.numel()),
                                              "first": int(d[0]), "last": int(d[-1])})
                # the same inputs again, long after they were written
                for r in (0, 1):
                    ys[r].fill_(0xAB)
                torch.cuda.synchronize()
                for r in (0, 1):
                    with torch.cuda.stream(streams[r]):
                        rts[r].gather(xs[r], ys[r])
                torch.cuda.synchronize()
                if not all(torch.equal(y, exp) for y in ys):
                    reuse_bad += 1
                    rd = []
                    for r in (0, 1):
                        d = (ys[r] != exp).nonzero().flatten()
                        if d.numel():
                            rd.append({"trial": t, "rank": r, "bytes_diff": int(d.numel()),
                                       "own": int((d // n == r).sum()), "first": int(d[0]), "last": int(d[-1]),
                                       "got_is_fill": bool((ys[r][d] == 0xAB).all()),
                                       "got_head": ys[r][d[:8]].tolist(), "want_head": exp[d[:8]].tolist()})
                    print("REUSE DIFF", json.dumps(rd), flush=True)
            row = {"bytes": n, "mode": mode, "trials": trials, "first_gather_bad": bad, "reuse_bad": reuse_bad,
                   "first_bad_detail": first}
            print(json.dumps(row), flush=True)
            if mode == "sync" and (bad or reuse_bad):
                ok_all = False
            if mode == "race" and reuse_bad:
                ok_all = False
    print("rank 0 stats:", json.dumps(rts[0].snapshot()), flush=True)
    torch.cuda.synchronize()
    for rt in rts:
        rt.close()
    print("VERDICT", "transport clean with synced inputs" if ok_all else "TRANSPORT MISMATCH", flush=True)


if __name__ == "__main__":
    main()
