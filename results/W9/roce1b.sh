#!/usr/bin/env bash
# W9: stage 2 again after the stress harness fix (allocations warmed per stream), then the extra size diagnostics
cd $HOME/glm53-tensorfold-spark
source <(sed -n '/^O=/,/^}/p' results/W9/roce1.sh)
dr stress-both 900 "" python tests/cuda/bench_roce.py stress --loop || { echo STOP | tee -a $O/SUMMARY; exit 1; }
dr stress-f0 900 "-e GLM53_TF_ROCE_HCA=rocep1s0f1" python tests/cuda/bench_roce.py stress --loop || { echo STOP | tee -a $O/SUMMARY; exit 1; }
dr stress-f1 900 "-e GLM53_TF_ROCE_HCA=roceP2p1s0f1" python tests/cuda/bench_roce.py stress --loop || { echo STOP | tee -a $O/SUMMARY; exit 1; }
dr stress-256k 900 "" python tests/cuda/bench_roce.py stress --loop --sizes 256k,1m --iters 20000
dr sizes-sync 1500 "" python w9/roce_sizes.py 256,512,1024,2048 300 sync
dr sizes-race2 900 "" python w9/roce_sizes.py 1024,2048 60 race
echo "roce single-node (b) done $(date +%T)" | tee -a $O/SUMMARY
