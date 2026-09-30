# OS tuning for a 2x DGX Spark (GB10) TP2 inference pair

The DGX OS image on a Spark is a desktop install: GDM, GNOME, printing, snaps, Bluetooth, modem and update daemons all
run on the box that serves the model. On a GB10, memory is unified, so everything the OS holds (including desktop GPU
buffers) comes out of the pool the engine and its KV cache need, and every OS wakeup can land on the cores that run the
engine's round loop.

This guide covers what we changed on both nodes of a TP2 pair (called **head** and **worker** below), why, the
measured effect, and how to apply and roll back each step with `scripts/os-tuning/`. Every change is reversible.
**These are our numbers on our rig. Measure your own before and after** (`measure.sh`, below).

## Results (measured, W20)

Same image and config in every load; decode / prefill with `bench/glmbench.py` (13 cells) and `bench/multiturn.py`
(4 concurrent streams, mean of 6 paired reps); memory = MemAvailable minimum during the 4 x 250k-token stress run.

| step | stress min head / worker (GiB) | 1 stream decode | 4 streams (tok/s) | prefill 24.5k / 98k (tok/s) |
| --- | --- | --- | --- | --- |
| A: as found (polkitd at 2.8 / 3.0 GiB after ~3 days of uptime) | 8.48 / **7.92** (below our 8 GiB gate) | baseline | 84.95 | 1,645 / 1,637 |
| B: + polkit restart and 512M cap (no reboot) | **11.18 / 10.90** | -0.1% (noise) | 84.93 | 1,650 / 1,645 |
| ALL: + headless, services, sysctl, IRQ/service pinning onto the A725 cores, journald (and MTU 9000), after a reboot | 10.72 / 10.77 | **+4.8%** | **88.58 (+4.3%)** | 1,659 / 1,654 (+0.5% vs B) |

- **Memory**: the stress floor went from 7.92 to ~10.7 GiB. Almost all of that is the polkit fix (step B alone: 8.48 /
  7.92 -> 11.18 / 10.90 GiB, decode unchanged). ALL is slightly below B because it ran straight after a reboot (nothing
  idle swapped out yet) and because of the MTU 9000 buffers (below).
- **Speed**: the +4.8% (1 stream) / +4.3% (4 streams) comes from the headless / services / IRQ-and-service-pinning set,
  not from the MTU (measured separately on the same boot). We applied that set as one batch and did not split it further.
  Outputs were bit-identical (greedy hashes 13/13, exactness checks, MMLU unchanged).
- **Reboot**: both nodes back on ssh in ~45 s, headless, `systemctl is-system-running` = `running`, no failed units.

Not adopted: pinning the server containers onto the X925 cores (`CPUSET=5-9,15-19` in `serve.sh`): **-1.5% at 4
streams** in every paired rep, because the HTTP threads then share the X925s with the engine. Leave `CPUSET` empty
(the default) unless you measure otherwise.

## The steps

All scripts run on the node itself as root, are idempotent, save the prior state in `/var/lib/spark-os-tuning/`
before changing anything, and accept `DRY_RUN=1` to print what they would do. Each `apply-0N` has a `rollback-0N`
that restores exactly what was recorded (a unit that was disabled before stays disabled).

| # | Script | What it changes | Why |
| --- | --- | --- | --- |
| 01 | `apply-01-polkit.sh` | restarts polkit; drop-in `MemoryMax=512M`, `MemorySwapMax=0`, `Restart=on-failure` | polkitd leaks (see below); frees ~2.5-3 GiB at once on a box that has been up a few days |
| 02 | `apply-02-headless.sh` | `set-default multi-user.target`; disables gdm and gnome-remote-desktop (`NOW=1` also stops gdm now) | the desktop costs RAM plus GPU memory (unified) and a steady stream of polkit / compositor wakeups |
| 03 | `apply-03-services.sh` | disables cups, avahi, bluetooth (+ rfkill), ModemManager, fwupd, snapd (refreshes held), multipathd (only if no mpath device), udisks2, upower, colord, switcheroo, accounts-daemon, rtkit, apport, motd / update-notifier / man-db timers. `TIER=B` adds rsyslog, lldpd, sysstat, apt timers. `EXTRA_KEEP="unit ..."` exempts units | unused on a headless server; ModemManager and snapd polling and snap refreshes are CPU noise |
| 04 | `apply-04-sysctl.sh` | `vm.dirty_background_bytes=256M`, `vm.dirty_bytes=1G`, `vm.stat_interval=10`, `kernel.sched_autogroup_enabled=0` | bounded dirty page cache instead of 10/20 % of RAM; fewer per-cpu vmstat wakeups; no desktop autogroups |
| 05 | `apply-05-cpu-irq.sh` | `CPUAffinity=0-4 10-14` drop-ins for ~30 chatty system services (live via `taskset`), `user.slice AllowedCPUs=0-4,10-14`, every movable IRQ and the default IRQ affinity to 0-4,10-14 at every boot (`spark-os-tuning.service`). `CPUIDLE=1` also disables the deepest idle state (LPI-3) on the X925s (experiment, not measured) | keeps OS noise on the Cortex-A725 cores and off the Cortex-X925 cores (5-9,15-19) the engine's hot threads use. docker / containerd are deliberately not pinned (containers inherit their affinity) |
| 06 | `apply-06-journald.sh` | `SystemMaxUse=1G` | frequent SSH sessions can fill several GB of journal in days |
| 07 | `apply-07-roce-mtu.sh` | CX7 link MTU 9000 (RoCE active_mtu 1024 -> 4096); `PERSIST=1 NETPLAN_FILE=...` writes it to netplan | optional, see below |

The cores: on GB10, cpu 0-4 and 10-14 are Cortex-A725 (`cpu_capacity` ~720), cpu 5-9 and 15-19 are Cortex-X925
(~1000). Check `cat /sys/devices/system/cpu/cpu*/cpu_capacity` on your box; override with `HOUSEKEEPING_CPUS` /
`ENGINE_CPUS` if different.

Not touched by any script: ssh, tailscaled, NetworkManager, wpa_supplicant and whatever interface carries your uplink
(keep it as it is), docker / containerd, the NVIDIA units (persistenced, clock / governor / NUMA units, telemetry,
dashboard), the RDMA stack (only 07 changes the MTU), rasdaemon, smartmontools, earlyoom, cron, user units.

## polkitd memory leak

On the stock image, polkitd grows by roughly **1 GiB a day per node** when something opens an SSH session per command
at a high rate. Our case: a monitoring dashboard polling each node over **Tailscale SSH** every 1-2 s. Tailscale SSH
creates a new logind session for every command channel. Each new session makes the desktop shell (the logged-in
session or even just the GDM greeter) call polkit `CheckAuthorization`, and polkitd leaks on each call. The same churn
wakes dbus, systemd, logind and journald hundreds of times a second.

Check yours: `ps -o rss=,etimes= -C polkitd` (RSS in KiB, uptime in s). Fixes, any of:

- `apply-01-polkit.sh`: restart it now and cap it (MemoryMax 512M; a fresh polkitd uses ~10 MiB). If the cap is hit,
  systemd restarts it within 2 s; polkit is D-Bus activated and its clients retry.
- Go headless (02). After our headless reboot polkitd stayed flat at 8-10 MiB for hours with the same polling running,
  so the cap became a backstop.
- Poll over a persistent connection: plain OpenSSH with `ControlMaster` / `ControlPersist` (one session per
  connection), an agent on the node, or a longer interval.

## CX7 MTU 9000 (optional)

Measured ~neutral for speed: +0.85% at 4 streams, within noise at 1 stream, mixed on NCCL micro-benchmarks. It costs
memory: **~0.75 GiB of MemFree per node idle** (mlx5 receive buffers) and **~1.4 GiB of MemAvailable per node with the
model loaded**. We kept it (stable, stress floor still 10.7 GiB); roll it back if memory is tight.

It must be applied on both nodes with the rank containers stopped (queue pairs take the MTU at creation):

```bash
scripts/serve.sh stop                                                   # on the head
sudo CX7_IFS="<cx7-if-1> <cx7-if-2>" scripts/os-tuning/apply-07-roce-mtu.sh          # each node
# persist: add PERSIST=1 NETPLAN_FILE=/etc/netplan/<file-that-defines-the-cx7-ports>.yaml
ping -M do -s 8972 -c 3 <worker-cx7-ip>                                 # from the head
```

## Apply

Worker first, then the head. Step 01 is safe at any time, even while serving.

```bash
cd /path/to/glm53-tensorfold-spark                 # on each node
sudo DRY_RUN=1 scripts/os-tuning/apply-all.sh      # preview
sudo scripts/os-tuning/apply-01-polkit.sh          # alone, if you only want the memory back now
sudo scripts/os-tuning/apply-all.sh                # 01-06 (not 07, not TIER=B, not CPUIDLE=1 unless exported)
sudo reboot                                        # in a planned window, worker first; completes the headless switch
```

`apply-all.sh` writes a `snapshot.sh` before and after into `/var/lib/spark-os-tuning/`. Useful options:
`LINGER_USER=<user>` on the head makes apply-02 refuse unless linger is on for the owner of your systemd user units
(for example a watchdog timer that restarts the server), so they still run without a login. apply-02 also refuses if
a Wi-Fi profile needs a desktop keyring (it prints the `nmcli` fix).

## Roll back

```bash
sudo scripts/os-tuning/rollback-all.sh             # every applied step, newest first
sudo scripts/os-tuning/rollback-0N-<name>.sh       # or one step
sudo reboot                                        # only needed to get the desktop back after rollback-02
```

`rollback-02-headless.sh START=1` starts gdm immediately; `systemctl isolate graphical.target` gives a one-off desktop
without changing anything.

## Reboot and power-loss recovery

- **BIOS "Restore on AC Power Loss"** must be set by hand in firmware setup on each box (Del during POST; usually under
  Advanced > Power / ACPI; set to Power On or Last State). There is no BMC and the setting is not readable or settable
  from Linux, so no script can do it. If the option is missing, update the firmware first (`fwupdmgr refresh &&
  fwupdmgr get-updates`, which works with fwupd.service disabled).
- **Headless boot chain**: NetworkManager joins the network with system profiles, tailscaled uses its stored key,
  docker starts, NVIDIA units run, and a watchdog / systemd unit that restarts the server brings serving back. Make
  sure the latter runs without a login (system unit, or user unit with linger) and that it handles the case where the
  rank containers are absent (not just exited), or a power loss while the server is stopped will not bring it back.
- **Old containers** with `restart=unless-stopped` / `always` come back after a power loss and take GPU memory; check
  `docker ps -a --format '{{.Names}} {{.Status}}'` and `docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' <name>`.
- **Check after a reboot** (on the head):

```bash
WORKER_SSH=<user>@<worker> CX7_IFS="<cx7-if-1> <cx7-if-2>" USER_TIMERS="<your-watchdog>.timer" \
  scripts/os-tuning/boot-check.sh --wait 30
```

It checks both nodes (multi-user target, no gdm, no failed units, default route, DNS, Tailscale if enabled, CX7
addresses, RoCE ports, docker, GPU persistence, the os-tuning unit, polkitd size, MemAvailable), then the server
(`:8000/v1/models`, both rank containers) and prints kernel-boot-to-serving seconds. Optional: `UPLINK_IF`,
`LINGER_USER`, `PUBLIC_URL`, `OLD_CONTAINERS` (regex of containers that must not be running).

Controlled test: record BEFORE, apply on both nodes, reboot the worker, wait for ssh, reboot the head, run
`boot-check.sh --wait 30`. Keep a console path (serial or monitor + keyboard) until the first headless boot works.
Optionally, with the BIOS setting done, cut AC to both boxes and check again.

## Measure it on your rig

```bash
# on the head, from the repo root, with the server running; same image and config each time
WORKER_SSH=<user>@<worker> MODEL=<served-model-name> scripts/os-tuning/measure.sh run BEFORE
#   ... apply on both nodes, reboot (worker first), server back up ...
WORKER_SSH=<user>@<worker> MODEL=<served-model-name> scripts/os-tuning/measure.sh run AFTER
scripts/os-tuning/measure.sh summary BEFORE AFTER            # results/os-tuning/summary-BEFORE-AFTER.md
```

A run takes OS snapshots of both nodes, 60 s of idle memory, a 1-stream decode probe (glmbench `tf`) and a 4-stream
probe (multiturn `concurrent`), each with a 30 s wakeup / `mpstat` sample on both nodes; `STRESS=1` adds the 4 x ~250k
memory stress with the MemAvailable minimum on both nodes. `snapshot.sh` alone gives a read-only view of one node:
memory by cgroup, polkitd, services, IRQ affinity, top wakeup sources.

Decode differences under ~1% are within rep-to-rep noise (4-stream sd ~3 tok/s); compare paired reps.

## Not recommended (for now)

- `nohz_full` / `isolcpus`: kernel command-line change; the engine is not a pure spin loop, gain uncertain.
- Removing swap or lowering swappiness: swap holds idle daemon memory that would otherwise count against MemAvailable.
- `THP=always`, disabling the soft-lockup watchdog, a global `CPUAffinity=` in `system.conf` (containers would inherit
  it and the engine would land on the A725s).

## Updates without the automatic updaters

fwupd and snap refreshes are off after 03 (apt timers too with `TIER=B`). Update by hand in a planned window, worker first:
`apt update && apt list --upgradable` (review driver and kernel packages), `apt upgrade`, reboot, `boot-check.sh`.
Snaps: `systemctl start snapd && snap refresh --unhold && snap refresh`.
