#!/usr/bin/env bash
# W16 step 1 (THEORY-2 §6 1a-1d): the no-server probes on ONE node, prod stopped, clocks locked by run.sh.
#   probes-node.sh head|worker [only-steps]
# head: CPU suites of 0510 / 0520 / 0530 in the image, littles (item 8), bench_decode_cold --mode both (item 7).
# worker: glprobe plain / under nsys graph / node tracing (item 2), the 0510 GPU test (split + lone: same bits),
#        hc_fused bits + microbench (item 3, when 0520's files exist).
# Every step is a fresh container under `timeout` (new kernels can hang). Logs: results/W16/probes-<node>/ (SUMMARY:
# one line a step). Works from either repo copy (head ~/..., worker $HOME/...).
node=$1; only=${2:-}
if [[ $node == head ]]; then cd $HOME/glm53-tensorfold-spark; else cd $HOME/glm53-tensorfold-spark; fi
O=$PWD/results/W16/probes-$node; mkdir -p "$O"
IMG=${IMAGE:-glm53-tensorfold:b8}
run() { # name timeout "docker args" cmd...
    local n=$1 t=$2 e=$3; shift 3
    if [[ -n "$only" && " $only " != *" $n "* ]]; then return 0; fi
    local t0=$(date +%s)
    docker rm -f w16-$n > /dev/null 2>&1
    docker run --rm --name w16-$n --gpus all --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK \
        --network host --ipc host -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 $e \
        -v "$PWD/tests:/work/tests" -v "$PWD/results/THEORY2-SESSION:/work/results/THEORY2-SESSION" -v "$O:/out" \
        --entrypoint bash "$IMG" -c "pip install -q pytest >/dev/null 2>&1; nvidia-smi --query-gpu=clocks.sm,clocks.mem,memory.used --format=csv,noheader; cd /work && PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda:/work/tests/cuda:/work/tests timeout -k 30 $t $*" \
        > "$O/$n.log" 2>&1
    local rc=$?
    docker rm -f w16-$n > /dev/null 2>&1
    echo "$(date +%T) $n rc=$rc $(( $(date +%s) - t0 ))s :: $(grep -E 'GATE|passed|failed|rror' "$O/$n.log" | tail -2 | tr '\n' ' ' | cut -c1-240)" | tee -a "$O/SUMMARY"
    return $rc
}
P="python -m pytest -q -p no:cacheprovider -rs"
T2=results/THEORY2-SESSION
echo "$(date +%T) start $node image $IMG" | tee -a "$O/SUMMARY"
if [[ $node == head ]]; then
    cpu="tests/test_batch_graphs_lone.py tests/test_http_pin.py tests/test_theory2_probes.py"
    for f in tests/test_hc_fused.py tests/test_hc_fused_compile.py; do [[ -f $f ]] && cpu="$cpu $f"; done
    run cpu 900 "" $P $cpu
    # item 8: pointer chase + bytes-in-flight sweep (littles.cu, nvcc in the image)
    run littles 1200 "" bash /work/$T2/probes/littles.sh /out ${LITTLES_ARGS:-}
    # item 7: 0440's dense kernel cold (rotation >= 96 MB, and behind a 64 MB streaming predecessor) vs old, per shape
    run cold 1200 "" python /work/tests/cuda/bench_decode_cold.py --mode both --json /out/cold.json ${COLD_ARGS:-}
else
    # item 2: graph launch host time + first-node delay, 100 / 400 / 1,650 nodes, split 2 / 4 / 8; plain, then under
    # nsys graph- and node-level tracing (the W11 inflation, measured)
    run glp-plain 600 "" python /work/$T2/probes/glprobe.py --tag plain --json /out/glprobe-plain.json
    run glp-graph 900 "--cap-add SYS_ADMIN" nsys profile --force-overwrite true --trace=cuda,nvtx --cuda-graph-trace=graph \
        -o /out/glp-graph python /work/$T2/probes/glprobe.py --tag nsys-graph --json /out/glprobe-graph.json
    run glp-node 900 "--cap-add SYS_ADMIN" nsys profile --force-overwrite true --trace=cuda,nvtx --cuda-graph-trace=node \
        -o /out/glp-node python /work/$T2/probes/glprobe.py --tag nsys-node --json /out/glprobe-node.json
    # 0510 on the GPU: split / probe main graphs and BATCH_GRAPHS=lone keep every bit (synthetic checkpoint)
    run t510 1800 "" $P -s tests/cuda/test_batch_graphs_lone_patches.py
    # item 3 (0520), when built: bits vs the Triton kernels, then the microbench (graph, cold L2, 50 MB predecessor)
    if [[ -f tests/cuda/test_hc_fused_patches.py ]]; then
        run t520 1800 "" $P -s tests/cuda/test_hc_fused_patches.py
        run hcbench 900 "" python /work/tests/cuda/bench_hc_fused.py --json /out/hcbench.json
    fi
fi
echo "$(date +%T) $node probes done" | tee -a "$O/SUMMARY"
