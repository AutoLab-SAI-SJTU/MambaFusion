#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

while true
do
    PORT=47579 # 47576
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT
export OMP_NUM_THREADS=24
python -m torch.distributed.launch --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch ${PY_ARGS}

