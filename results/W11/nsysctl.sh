#!/usr/bin/env bash
# W11: nsysctl.sh start NAME | stop  -- start / stop the nsys collection on both ranks (session w11). ONCE per server start.
W=$WORKER_SSH
case $1 in
  start)
    docker exec glm53-tf-r0 nsys start --session=w11 --sample=none --cpuctxsw=none --force-overwrite=true -o /w11/out/$2-r0 &
    ssh -o BatchMode=yes $W docker exec glm53-tf-r1 nsys start --session=w11 --sample=none --cpuctxsw=none --force-overwrite=true -o /w11/out/$2-r1 &
    wait ;;
  stop)
    docker exec glm53-tf-r0 nsys stop --session=w11 &
    ssh -o BatchMode=yes $W docker exec glm53-tf-r1 nsys stop --session=w11 &
    wait ;;
esac
