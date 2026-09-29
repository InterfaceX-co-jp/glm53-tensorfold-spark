#!/usr/bin/env bash
# W7: nsysctl.sh start NAME | stop  -- start / stop the nsys collection on both ranks (session w7)
W=$WORKER_SSH
case $1 in
  start)
    docker exec glm53-tf-r0 nsys start --session=w7 --sample=none --cpuctxsw=none --force-overwrite=true -o /w7/out/$2-r0 &
    ssh -o BatchMode=yes $W docker exec glm53-tf-r1 nsys start --session=w7 --sample=none --cpuctxsw=none --force-overwrite=true -o /w7/out/$2-r1 &
    wait ;;
  stop)
    docker exec glm53-tf-r0 nsys stop --session=w7 &
    ssh -o BatchMode=yes $W docker exec glm53-tf-r1 nsys stop --session=w7 &
    wait ;;
esac
