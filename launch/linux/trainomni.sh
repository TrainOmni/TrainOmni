#!/usr/bin/env sh
set -eu

if [ -z "${TRAINOMNI_PYTHON:-}" ]; then
    echo 'TRAINOMNI_PYTHON must be an absolute path to the intended Python executable.' >&2
    exit 2
fi

case "$TRAINOMNI_PYTHON" in
    /*) ;;
    *)
        echo 'TRAINOMNI_PYTHON must be an absolute path.' >&2
        exit 2
        ;;
esac

if [ ! -f "$TRAINOMNI_PYTHON" ] || [ ! -x "$TRAINOMNI_PYTHON" ]; then
    echo "TRAINOMNI_PYTHON is not an executable file: $TRAINOMNI_PYTHON" >&2
    exit 2
fi

exec "$TRAINOMNI_PYTHON" -m trainomni "$@"
