#!/usr/bin/env bash
# W11: nsys-rep -> sqlite for NAME (both ranks; rank 1's report copied from the worker node), in a throwaway b4 container
N=$1; O=/var/tmp/w11/out
scp -q $WORKER_SSH:$O/$N-r1.nsys-rep $O/
docker run --rm --name w11-export-$N --entrypoint bash -v $O:/o glm53-tensorfold:b4 -c "for r in r0 r1; do nsys export --type sqlite -f true -o /o/$N-\$r.sqlite /o/$N-\$r.nsys-rep > /dev/null 2>&1; done; ls -la /o"
