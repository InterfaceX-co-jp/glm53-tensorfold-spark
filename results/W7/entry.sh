#!/usr/bin/env bash
# W7: container entrypoint wrapper -- the image's entrypoint with /w7/bin first on PATH, so its final
# `exec tensorfold ...` runs under `nsys launch` (W7_NSYS=1): the process starts inside an idle nsys session
# "w7"; `nsys start --session=w7` / `nsys stop --session=w7` (docker exec) collect only the chosen windows.
export PATH=/w7/bin:$PATH
exec /usr/local/bin/glm53-tf-entrypoint "$@"
