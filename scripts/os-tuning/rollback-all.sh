#!/usr/bin/env bash
# Roll back every os-tuning step applied on this node, newest first.   sudo scripts/os-tuning/rollback-all.sh
source "$(dirname "$0")/lib.sh"; need_root
here=$(cd "$(dirname "$0")" && pwd)
[[ -f $(state_file 07-roce-mtu) ]] && bash "$here/rollback-07-roce-mtu.sh"
[[ -f /etc/systemd/journald.conf.d/90-spark.conf ]] && bash "$here/rollback-06-journald.sh"
[[ -f $(state_file 05-cpu-irq) ]] && bash "$here/rollback-05-cpu-irq.sh"
[[ -f $(state_file 04-sysctl) ]] && bash "$here/rollback-04-sysctl.sh"
[[ -f $(state_file 03-services) ]] && bash "$here/rollback-03-services.sh"
[[ -f $(state_file 02-headless) || -f $(state_file 02-headless-default) ]] && bash "$here/rollback-02-headless.sh"
[[ -f /etc/systemd/system/polkit.service.d/90-spark-memory.conf ]] && bash "$here/rollback-01-polkit.sh"
log "rollback-all done; reboot to return to the graphical target if 02 was rolled back"
