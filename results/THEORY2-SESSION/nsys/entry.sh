#!/usr/bin/env bash
# W16 (W11's): container entrypoint wrapper -- the image's entrypoint with /w16/bin first on PATH, so its final
# `exec tensorfold ...` runs under `nsys launch` (W16_NSYS=1) in an idle session "w16"; `nsys start/stop --session=w16`
# (docker exec, run.sh nsysctl) collect ONE window per server start (a second start/stop cycle took both ranks down in W7).
export PATH=/w16/bin:$PATH
exec /usr/local/bin/glm53-tf-entrypoint "$@"
