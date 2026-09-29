#!/usr/bin/env bash
# W11 window 1 (prod stopped): one-GPU probes in throwaway containers (name w11-*), both nodes.
#   bench.sh probe   probe.cu on the head node and the worker node at once (bandwidth ceiling, launch overhead, PDL)
#   bench.sh kbench  kbench.py (production decode kernels, random weights) on the head node
#   bench.sh ncu     ncu of kbench --ncu (grouped_kernel, _qmm): dram bytes / throughput, occupancy, launch stats
#   bench.sh gm      nsys --gpu-metrics-devices on probe --quick (is DRAM-throughput sampling available on GB10?)
cd $HOME/glm53-tensorfold-spark
R=results/W11; O=/var/tmp/w11/bench; mkdir -p $O
IMG=glm53-tensorfold:b4
RUN="docker run --rm --gpus all --ipc=host --cap-add SYS_ADMIN --entrypoint bash -v $HOME/glm53-tensorfold-spark:/work -v $O:/o -w /work"
case $1 in
  probe)
    $RUN --name w11-probe $IMG -c "nvcc -O3 -arch=sm_121 -o /o/probe $R/probe.cu && /o/probe" > $R/probe-head.log 2>&1 &
    ssh -o BatchMode=yes $WORKER_SSH "mkdir -p /var/tmp/w11/bench"
    scp -q $R/probe.cu $WORKER_SSH:/var/tmp/w11/bench/
    ssh -o BatchMode=yes $WORKER_SSH "docker run --rm --gpus all --ipc=host --name w11-probe --entrypoint bash -v /var/tmp/w11/bench:/o $IMG -c 'nvcc -O3 -arch=sm_121 -o /o/probe /o/probe.cu && /o/probe'" > $R/probe-worker.log 2>&1 &
    wait; tail -3 $R/probe-head.log $R/probe-worker.log ;;
  kbench)
    $RUN --name w11-kbench -e PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda $IMG -c "python3 $R/kbench.py --out /o/kbench.json" > $R/kbench.log 2>&1
    cp $O/kbench.json $R/kbench.json; tail -5 $R/kbench.log ;;
  ncu)
    $RUN --name w11-ncu -e PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda $IMG -c "ncu --target-processes all --kernel-name regex:'^(grouped_kernel|_qmm)\$' --launch-count 400 --section SpeedOfLight --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis --metrics dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,launch__grid_size,sm__warps_active.avg.pct_of_peak_sustained_active -f -o /o/kb python3 $R/kbench.py --ncu && ncu -i /o/kb.ncu-rep --page raw --csv > /o/kb-raw.csv" > $R/ncu.log 2>&1
    cp $O/kb-raw.csv $R/ncu-kb-raw.csv; tail -5 $R/ncu.log ;;
  gm)
    $RUN --name w11-gm $IMG -c "nvcc -O3 -arch=sm_121 -o /o/probe $R/probe.cu && nsys profile --gpu-metrics-devices=all --gpu-metrics-frequency=20000 -t cuda -f true -o /o/gm /o/probe --quick > /o/gm.out 2>&1; tail -5 /o/gm.out; nsys export --type sqlite -f true -o /o/gm.sqlite /o/gm.nsys-rep > /dev/null 2>&1; ls -la /o" > $R/gm.log 2>&1
    tail -12 $R/gm.log ;;
esac
