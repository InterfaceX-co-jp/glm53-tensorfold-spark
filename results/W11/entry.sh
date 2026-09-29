#!/usr/bin/env bash
# W11 (W7's): container entrypoint wrapper -- the image's entrypoint with /w11/bin first on PATH, so its final
# `exec tensorfold ...` runs under `nsys launch` (W11_NSYS=1) in an idle session "w11"; `nsys start/stop --session=w11`
# (docker exec, nsysctl.sh) collect ONE window per server start (a second start/stop cycle took both ranks down in W7).
export PATH=/w11/bin:$PATH
exec /usr/local/bin/glm53-tf-entrypoint "$@"
