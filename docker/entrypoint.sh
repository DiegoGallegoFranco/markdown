#!/bin/sh
# Punto de entrada único: `web` levanta el servidor, cualquier otra cosa se
# pasa tal cual (para correr la CLI dentro del contenedor).
#
#   docker compose up -d                                    # servidor web
#   docker compose run --rm app cli inventario /data/entrada
#   docker compose run --rm app python -m pipeline qc --output /data/salida
set -eu

mkdir -p "${DATA_DIR:-/data}"

case "${1:-web}" in
  web)
    exec uvicorn webapp.main:app \
        --host "${HOST:-0.0.0.0}" \
        --port "${PORT:-8000}" \
        --workers 1 \
        --timeout-keep-alive 75
    ;;
  cli)
    shift
    exec python -m pipeline "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
