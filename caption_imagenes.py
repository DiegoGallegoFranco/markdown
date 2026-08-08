#!/usr/bin/env python3
"""Genera descripciones (captions) de las imágenes extraídas, vía API de Claude.

Wrapper de `python -m pipeline captions`. Es REANUDABLE: las imágenes que ya
tienen caption se saltan sin llamar a la API, así que volver a ejecutarlo tras
un fallo a mitad de lote no repaga nada ni duplica descripciones.

Uso:
    python caption_imagenes.py                       # todo output/
    python caption_imagenes.py --dry-run             # solo cuenta
    python caption_imagenes.py --limit 3             # smoke test
    python caption_imagenes.py --from-file f.txt     # solo las imágenes listadas
"""
import sys

from pipeline.cli import main

if __name__ == "__main__":
    sys.exit(main(["captions", *sys.argv[1:]]))
