#!/bin/bash
# Convierte DOCX a Markdown con pandoc, extrayendo las imágenes embebidas.
#
# Wrapper de `python -m pipeline convertir`. Acepta rutas absolutas fuera del
# repo (la versión anterior en bash puro generaba output//Users/... con ellas).
#
# Requiere: brew install pandoc (o el gestor de paquetes del SO).
#
# Nota: pandoc solo extrae imágenes referenciadas en el cuerpo del documento
# (no las de encabezados/pies). Si --extract-media saca imágenes que no
# aparecen enlazadas en el .md, son decorativas.
#
# Uso:
#   ./convertir_docx.sh "carpeta/documento.docx" ["otro.docx" ...]

set -uo pipefail

if [ "$#" -eq 0 ]; then
  echo "Uso: $0 archivo1.docx [archivo2.docx ...]" >&2
  exit 1
fi

cd "$(dirname "$0")" || exit 1
exec python -m pipeline convertir "$@"
