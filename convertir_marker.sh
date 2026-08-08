#!/bin/bash
# Convierte PDFs a Markdown con Marker (layout VLM + OCR completo).
#
# Wrapper de `python -m pipeline convertir --engine marker`. La lógica (aplanado
# de la subcarpeta que anida marker_single, timeout, log por documento, seguir
# el lote tras un error) vive en pipeline/converters.py, compartida con la web.
#
# Requiere: pip install -r requirements.txt, y en Mac sin GPU NVIDIA:
#   brew install llama.cpp
# IMPORTANTE: aplicar antes patches/fix_surya_grammar.py y
# patches/fix_marker_empty_image.py o la conversión se degrada en silencio.
#
# Uso:
#   ./convertir_marker.sh "carpeta/documento.pdf" ["otro.pdf" ...]

set -uo pipefail

if [ "$#" -eq 0 ]; then
  echo "Uso: $0 archivo1.pdf [archivo2.pdf ...]" >&2
  exit 1
fi

cd "$(dirname "$0")" || exit 1
exec python -m pipeline convertir --engine marker "$@" 2>&1 | tee -a marker_batch.log
