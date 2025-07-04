# #!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}
while true
do
    PORT=37576 # 47576
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

python -m torch.distributed.launch --nproc_per_node=${NGPUS}  --rdzv_endpoint=localhost:${PORT} test.py --launcher pytorch ${PY_ARGS}

# #!/usr/bin/env bash

# set -x
# NGPUS=$1
# PORT=${2:-37576}   # Use the second argument as PORT, or default to 37576
# PY_ARGS=${@:3}      # Additional arguments start from the third argument

# while true
# do
#     status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
#     if [ "${status}" != "0" ]; then
#         break
#     fi
# done

# python -m torch.distributed.launch --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} test.py --launcher pytorch ${PY_ARGS}
