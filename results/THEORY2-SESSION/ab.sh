#!/usr/bin/env bash
# W13 per-load set (W12's ab.sh, trimmed per THEORY-2 §6): ab.sh TAG [full|short]
#   gates : warm-up, exact 10/10 (drafted == serial), batchexact 4/4 (batched == alone), transcripts (== load C's,
#           and == W9 A when results/W9/transcripts-A.log exists), ab.py 24.5k / 98k once (reply sha
#           8794a3463259cc2f + prefill tok/s), every glmbench reply hash == load C's (summary.py);
#   speed : glmbench tf,kit,edit x3 (1 stream), concurrent 4 streams x3 twice (6 reps), lone requests over slots 0-3
#           (slots.py);  short: glmbench tf,kit x1 + concurrent x3 once (the CT trace load);
#   then /health and the engine's error lines.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
cd "$REPO"
tag=$1; kind=${2:-full}
SUITES=${GLMB_SUITES:-tf,kit,edit}
log "ab $tag ($kind) start"
python3 results/W7/req.py prefill "$R/warm-$tag.jsonl" 12000 8 '{}' > /dev/null 2>&1
if [[ $kind == full ]]; then
    python3 bench/glmbench.py --base $B --model $M --suites exact --out "$R/exact-$tag.json" > "$R/exact-$tag.log" 2>&1
    python3 bench/multiturn.py --base $B --model $M --modes batchexact --out "$R/batchexact-$tag.json" > "$R/batchexact-$tag.log" 2>&1
    log "$tag exact: $(grep -c 'identical=True' "$R/exact-$tag.log")/$(grep -c 'identical=' "$R/exact-$tag.log") | $(grep -E 'batched == alone' "$R/batchexact-$tag.log" | tail -1)"
    python3 results/W10/transcripts.py "$R/transcripts-$tag.json" > "$R/transcripts-$tag.log" 2>&1
    python3 - "$R/transcripts-$tag.log" "$R/transcripts-C.log" results/W9/transcripts-A.log <<'PY' | tee -a "$R/session.log" >&2
import os, sys
def first(p):
    return open(p).read().splitlines()[0] if os.path.exists(p) and os.path.getsize(p) else None
me, c, a = (first(p) for p in sys.argv[1:4])
print("transcripts:", "alone == C" if me and c and me == c else ("C itself" if sys.argv[1] == sys.argv[2] else f"alone vs C: {'SAME' if me == c else 'DIFFER'}"),
      "| == W9 A:", (me == a) if a else "n/a (no W9 log)")
PY
    python3 results/W5/ab.py "$R/ab-$tag.json" 24500,98000 '{"prod":{}}' > "$R/ab-$tag.log" 2>&1
    log "$tag ab.py: $(grep -o 'sha [0-9a-f/]*' "$R/ab-$tag.log" | tr '\n' ' ') (want 8794a3463259cc2f)"
    python3 bench/glmbench.py --base $B --model $M --suites "$SUITES" --reps 3 --long-tokens 512 --label "$tag" \
        --out "$R/glmbench-$tag.json" > "$R/glmbench-$tag.log" 2>&1
    for pass in a b; do
        python3 bench/multiturn.py --base $B --model $M --modes concurrent --streams 4 --reps 3 --long-tokens 512 \
            --out "$R/conc-$tag-$pass.json" > "$R/conc-$tag-$pass.log" 2>&1
        log "$tag conc $pass: $(grep -oE 'aggregate [0-9.]+ tok/s' "$R/conc-$tag-$pass.log" | tr '\n' ' ')"
    done
    python3 "$T2/slots.py" "$R/slots-$tag.json" > "$R/slots-$tag.log" 2>&1
    log "$tag $(tail -1 "$R/slots-$tag.log")"
else
    python3 bench/glmbench.py --base $B --model $M --suites tf,kit --reps 1 --label "$tag" \
        --out "$R/glmbench-$tag.json" > "$R/glmbench-$tag.log" 2>&1
    python3 bench/multiturn.py --base $B --model $M --modes concurrent --streams 4 --reps 3 --long-tokens 512 \
        --out "$R/conc-$tag-a.json" > "$R/conc-$tag-a.log" 2>&1
    log "$tag conc a: $(grep -oE 'aggregate [0-9.]+ tok/s' "$R/conc-$tag-a.log" | tr '\n' ' ')"
fi
curl -s -m 10 $B/health > "$R/health-$tag.json"
log "$tag health: $(head -c 300 "$R/health-$tag.json") | r0 error lines: $(docker logs glm53-tf-r0 2>&1 | grep -ciE 'traceback|error')"
log "ab $tag done"
