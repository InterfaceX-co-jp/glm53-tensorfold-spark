#!/usr/bin/env bash
# Read-only snapshot of the OS state docs/OS-TUNING.md cares about. Changes nothing; safe while serving.
#   sudo scripts/os-tuning/snapshot.sh [OUTFILE]          (default: stdout; works without root, with less detail)
#   WAKE_S=5   seconds for the wakeup sample (per-thread context switches + /proc/interrupts deltas); 0 = skip
#   CX7_IFS="<if1> <if2>"   optional: print their MTU
# Run it before apply-*, after apply-* and after the reboot test; diff the files. It can be piped to the worker:
#   ssh "$WORKER" 'WAKE_S=5 bash -s' < scripts/os-tuning/snapshot.sh > os-worker.txt
set -uo pipefail
out=${1:-/dev/stdout}
WAKE_S=${WAKE_S:-5}
{
echo "# os-tuning snapshot $(hostname -s) $(date -Is) uptime_s=$(cut -d' ' -f1 /proc/uptime)"
echo "## meminfo (GiB)"
awk '/^(MemTotal|MemFree|MemAvailable|Cached|Shmem|SUnreclaim|AnonPages|SwapTotal|SwapFree):/{printf "%s %.2f\n",$1,$2/1048576}' /proc/meminfo
echo "## default target: $(systemctl get-default)"
echo "## sessions"; loginctl list-sessions --no-legend 2>/dev/null | awk '{print $1,$2,$3,$4}'
echo "## polkitd"; ps -o pid=,etimes=,rss=,time= -C polkitd 2>/dev/null | awk '{printf "pid %s up %.1f h rss %.2f GiB cpu %s\n",$1,$2/3600,$3/1048576,$4}'
grep VmSwap "/proc/$(pgrep -xo polkitd 2>/dev/null)/status" 2>/dev/null
echo "## top cgroups by memory (MiB)"
for d in /sys/fs/cgroup/system.slice/*.service /sys/fs/cgroup/system.slice/docker-*.scope /sys/fs/cgroup/user.slice/user-*.slice; do
  [[ -r $d/memory.current ]] && echo "$(( $(cat "$d"/memory.current) / 1048576 )) ${d#/sys/fs/cgroup/}"
done | sort -rn | head -25
echo "## GPU memory per process"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1
nvidia-smi 2>/dev/null | awk '/ G  /{print "graphics:",$5,$(NF-1)}'
echo "## running services"; systemctl list-units --type=service --state=running --no-legend --no-pager | awk '{print $1}' | tr '\n' ' '; echo
echo "## timers"; systemctl list-timers --no-legend --no-pager | awk '{print $(NF-1)}' | tr '\n' ' '; echo
echo "## sysctl"; sysctl vm.swappiness vm.dirty_ratio vm.dirty_background_ratio vm.dirty_bytes vm.dirty_background_bytes \
  vm.stat_interval kernel.sched_autogroup_enabled kernel.numa_balancing kernel.watchdog kernel.nmi_watchdog 2>&1 | tr '\n' ';'; echo
echo "## cpufreq"; for c in 0 5; do echo "cpu$c $(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null) $(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq 2>/dev/null)"; done
echo "## cpuidle disabled states (cpu:state)"; for s in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do [[ $(cat "$s") == 1 ]] && echo "$s"; done | sed -E 's#.*/cpu([0-9]+)/cpuidle/state([0-9])/.*#\1:\2#' | tr '\n' ' '; echo
echo "## IRQ affinity (non-per-cpu)"; for i in /proc/irq/[0-9]*; do n=${i##*/}; name=$(awk -v n="$n:" '$1==n{print $NF}' /proc/interrupts); [[ -z "$name" ]] && continue
  case "$name" in *mlx5_comp*|*nvme*q*) continue;; esac; echo "$n $(cat "$i"/smp_affinity_list 2>/dev/null) eff=$(cat "$i"/effective_affinity_list 2>/dev/null) $name"; done | head -60
echo "## network"; ip -4 route show default | sed -E 's/via [0-9.]+/via <gw>/'; ip -br link | grep -vE '^(lo|docker0|br-|veth)'
for i in ${CX7_IFS:-}; do echo "$i mtu=$(cat "/sys/class/net/$i/mtu" 2>/dev/null)"; done
echo "## docker"; docker ps --format '{{.Names}} {{.Image}} {{.Status}}' 2>&1
if [[ "$WAKE_S" != 0 ]]; then
  echo "## wakeups over ${WAKE_S}s: per-thread context switches (top 25) and interrupts"
  snap() { for t in /proc/[0-9]*/task/[0-9]*; do awk -v t="$t" '/^voluntary_ctxt_switches|^nonvoluntary_ctxt_switches/{s+=$2}END{print t, s}' "$t/status" 2>/dev/null; done | sort; }
  a=$(mktemp); b=$(mktemp); ia=$(mktemp); ib=$(mktemp)
  snap > "$a"; cat /proc/interrupts > "$ia"; c0=$(awk '/^ctxt/{print $2}' /proc/stat)
  sleep "$WAKE_S"
  snap > "$b"; cat /proc/interrupts > "$ib"; c1=$(awk '/^ctxt/{print $2}' /proc/stat)
  echo "system context switches/s: $(( (c1 - c0) / WAKE_S ))"
  join "$a" "$b" | awk -v s="$WAKE_S" '{d=($3-$2)/s; if(d>=1) printf "%.0f %s\n", d, $1}' | sort -rn | head -25 | while read -r d t; do
    p=${t#/proc/}; p=${p%%/*}; echo "$d/s $(cat "$t/comm" 2>/dev/null) pid=$p $(awk -F: 'NR==1{print $3}' /proc/"$p"/cgroup 2>/dev/null)"; done
  echo "interrupts/s (top 15, all cpus):"
  isum='NR>1{s=0; for(i=2;i<=NF;i++){if($i!~/^[0-9]+$/) break; s+=$i}; print $1, s, $NF}'
  paste <(awk "$isum" "$ia") <(awk "$isum" "$ib" | awk '{print $2, $3}') \
    | awk -v s="$WAKE_S" '{d=($4-$2)/s; if(d>0) printf "%.0f %s %s\n", d, $1, $5}' | sort -rn | head -15
  rm -f "$a" "$b" "$ia" "$ib"
fi
} > "$out"
