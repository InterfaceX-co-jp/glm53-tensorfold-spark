#!/usr/bin/env bash
# W13 (W11's): container entrypoint wrapper -- the image's entrypoint with /w13/bin first on PATH, so its final
# `exec tensorfold ...` runs under `nsys launch` (W13_NSYS=1) in an idle session "w13"; `nsys start/stop --session=w13`
# (docker exec, run.sh nsysctl) collect ONE window per server start (a second start/stop cycle took both ranks down in W7).
export PATH=/w13/bin:$PATH
exec /usr/local/bin/glm53-tf-entrypoint "$@"
