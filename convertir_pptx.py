#!/usr/bin/env python3
"""Convierte PPTX a Markdown (texto por diapositiva, tablas, notas e imágenes).

Wrapper de `python -m pipeline convertir`.

Uso:
    python convertir_pptx.py "carpeta/presentacion.pptx" ["otra.pptx" ...]
"""
import sys

from pipeline.cli import main

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python convertir_pptx.py archivo1.pptx [archivo2.pptx ...]", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(["convertir", *sys.argv[1:]]))
