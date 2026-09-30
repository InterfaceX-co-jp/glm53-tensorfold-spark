#!/usr/bin/env bash
# Boot-time part of 05 (installed as /usr/local/sbin/spark-os-tuning-runtime.sh, run by spark-os-tuning.service).
# Moves every movable IRQ to the housekeeping cores and, with CPUIDLE=1 in /etc/default/spark-os-tuning, disables the
# deepest idle state (LPI-3, ~433 us exit latency on GB10) on the engine cores.
# Managed IRQs (mlx5_comp*, NVMe queues: one per cpu, kernel-owned) refuse the write and keep their per-cpu spread.
set -uo pipefail
[[ -f /etc/default/spark-os-tuning ]] && . /etc/default/spark-os-tuning
HK=${HOUSEKEEPING_CPUS:-0-4,10-14}; EN=${ENGINE_CPUS:-5-9,15-19}; CPUIDLE=${CPUIDLE:-0}
mask() { # cpu list -> hex mask
    local m=0 part a b i parts
    IFS=, read -ra parts <<< "$1"
    for part in "${parts[@]}"; do a=${part%-*}; b=${part#*-}; for ((i=a; i<=b; i++)); do m=$((m | (1 << i))); done; done
    printf '%x' $m
}
expand() { local part a b i parts; IFS=, read -ra parts <<< "$1"; for part in "${parts[@]}"; do a=${part%-*}; b=${part#*-}; for ((i=a; i<=b; i++)); do echo $i; done; done; }
mask "$HK" > /proc/irq/default_smp_affinity 2>/dev/null
moved=0; kept=0
for d in /proc/irq/[0-9]*; do
    [[ -w $d/smp_affinity_list ]] || continue
    if echo "$HK" > "$d/smp_affinity_list" 2>/dev/null; then moved=$((moved+1)); else kept=$((kept+1)); fi
done
echo "irq: $moved moved to $HK, $kept managed / refused"
if [[ "$CPUIDLE" == 1 ]]; then
    for c in $(expand "$EN"); do
        for s in /sys/devices/system/cpu/cpu"$c"/cpuidle/state*; do
            [[ "$(cat "$s"/name)" == LPI-3 ]] && echo 1 > "$s"/disable
        done
    done
    echo "cpuidle: LPI-3 disabled on $EN"
fi
