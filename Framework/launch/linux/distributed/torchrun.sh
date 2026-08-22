#!/usr/bin/env sh
set -eu

NPROC_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

while [ "$#" -gt 0 ]; do
    case "$1" in
        --nproc-per-node) NPROC_PER_NODE=$2; shift 2 ;;
        --nnodes) NNODES=$2; shift 2 ;;
        --node-rank) NODE_RANK=$2; shift 2 ;;
        --master-addr) MASTER_ADDR=$2; shift 2 ;;
        --master-port) MASTER_PORT=$2; shift 2 ;;
        --) shift; break ;;
        *)
            echo 'distributed launcher options must end with -- before TrainOmni arguments' >&2
            exit 2
            ;;
    esac
done

if [ -z "${TRAINOMNI_PYTHON:-}" ]; then
    echo 'TRAINOMNI_PYTHON must be an absolute path to the intended Python executable.' >&2
    exit 2
fi
case "$TRAINOMNI_PYTHON" in
    /*) ;;
    *) echo 'TRAINOMNI_PYTHON must be an absolute path.' >&2; exit 2 ;;
esac
if [ ! -f "$TRAINOMNI_PYTHON" ] || [ ! -x "$TRAINOMNI_PYTHON" ]; then
    echo "TRAINOMNI_PYTHON is not an executable file: $TRAINOMNI_PYTHON" >&2
    exit 2
fi
if [ "$NODE_RANK" -ge "$NNODES" ]; then
    echo 'node rank must be less than node count' >&2
    exit 2
fi

exec "$TRAINOMNI_PYTHON" -m torch.distributed.run \
    "--nproc-per-node=$NPROC_PER_NODE" \
    "--nnodes=$NNODES" \
    "--node-rank=$NODE_RANK" \
    "--master-addr=$MASTER_ADDR" \
    "--master-port=$MASTER_PORT" \
    -m trainomni "$@"
