#!/usr/bin/env bash
# W9 window 2: A = prod config (NCCL control; RoCE A/B side + item 5 serial rounds), C = A + GLM53_TF_COMM_BACKEND=roce,
# B = A + GLM53_TF_B12X=4 (0360). Each on the b2 image from config/prod.env.
cd $HOME/glm53-tensorfold-spark
R=results/W9
bash $R/mem.sh > $R/mem-w2.log 2>&1 & echo $! > $R/mem.pid
bash $R/roce2d.sh > $R/roce2d.out 2>&1
rmmark() { for h in local $WORKER_SSH; do c="docker run --rm -v glm53-tf-cache:/cache --entrypoint bash glm53-tensorfold:b2 -c 'cat /cache/roce-failed 2>/dev/null; rm -f /cache/roce-failed'"; if [ $h = local ]; then eval "$c"; else ssh -o BatchMode=yes $h "$c"; fi; done; }
rmmark > $R/roce-marks-before-C.txt 2>&1
bash $R/load.sh A
bash $R/runAC.sh A perslot > $R/runA.out 2>&1
rmmark >> $R/roce-marks-before-C.txt 2>&1
bash $R/load.sh C GLM53_TF_COMM_BACKEND=roce
bash $R/runAC.sh C > $R/runC.out 2>&1
bash $R/load.sh B GLM53_TF_B12X=4
bash $R/runB.sh B > $R/runB.out 2>&1
echo "w2 done $(date +%T)" | tee -a $R/loads.log
