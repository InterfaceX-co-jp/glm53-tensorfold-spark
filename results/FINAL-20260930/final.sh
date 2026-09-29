#!/usr/bin/env bash
# W17 phase 2: final numbers on the final production config (prod serving, no restart), 3 rounds:
#   RigMark standard suite (new COMPARISON_ID each round) -> rigmark-rN/ (+ results/rigmark/tensorfold-20260930-w17-final-rN)
#   glmbench tf,kit,edit x3 -> glmbench-rN.json; concurrent 1 / 4 streams x3 -> conc1-rN / conc4-rN; ab.py 24.5k / 98k -> ab-rN
# then MMLU-200 once (quality.json), exact / batchexact once. summarize.py -> summary.md. final.sh [FIRST_ROUND] [LAST_ROUND]
cd $HOME/glm53-tensorfold-spark
F=results/FINAL-20260930; B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3
step() { echo "=== $* ($(date +%T))"; }
for r in $(seq ${1:-1} ${2:-3}); do
  step "round $r rigmark"
  before=$(ls -d results/rigmark/tensorfold-2* 2>/dev/null | sort | tail -1)
  COMPARISON_ID=2026-09-glm53-exl3-2xspark-tensorfold-w17-final-r$r scripts/rigmark/run.sh tensorfold > $F/rigmark-r$r.out 2>&1; echo "rigmark rc=$?"
  d=$(ls -d results/rigmark/tensorfold-2* | sort | tail -1)
  if [[ "$d" != "$before" ]]; then mv "$d" results/rigmark/tensorfold-20260930-w17-final-r$r; cp -r results/rigmark/tensorfold-20260930-w17-final-r$r $F/rigmark-r$r; fi
  tail -3 $F/rigmark-r$r.out | cut -c1-200
  step "round $r glmbench"
  python3 bench/glmbench.py --base $B --model $M --suites tf,kit,edit --reps 3 --long-tokens 512 --label final-r$r --out $F/glmbench-r$r.json > $F/glmbench-r$r.log 2>&1
  grep -E 'median' $F/glmbench-r$r.log | cut -c1-80
  for s in 1 4; do
    python3 bench/multiturn.py --base $B --model $M --modes concurrent --streams $s --reps 3 --long-tokens 512 --out $F/conc$s-r$r.json > $F/conc$s-r$r.log 2>&1
    grep -E 'streams rep' $F/conc$s-r$r.log | cut -c1-60
  done
  step "round $r prefill"
  python3 results/W5/ab.py $F/ab-r$r.json 24500,98000 '{"prod":{}}' > $F/ab-r$r.log 2>&1; cut -c1-110,200- $F/ab-r$r.log | tail -2
done
if [[ "${2:-3}" == 3 ]]; then
  step MMLU; python3 bench/quality.py --base $B --model $M --label final --out $F/quality.json > $F/quality.log 2>&1; tail -3 $F/quality.log
  step exact
  python3 bench/glmbench.py --base $B --model $M --suites exact --out $F/exact.json > $F/exact.log 2>&1
  python3 bench/multiturn.py --base $B --model $M --modes batchexact --out $F/batchexact.json > $F/batchexact.log 2>&1
  echo "exact: $(tail -1 $F/exact.log | grep -o true | wc -l)/10"; grep -E 'batched == alone' $F/batchexact.log
fi
curl -s $B/health; echo; bash results/W17/slotcheck.sh
step done
