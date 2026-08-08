#!/usr/bin/env python3
"""Convierte PDFs nativos a Markdown con pymupdf4llm (rápido, solo CPU).

Wrapper de `python -m pipeline convertir --engine pymupdf`. Acepta rutas
relativas o absolutas, dentro o fuera de este repo.

Uso:
    python convertir_pymupdf.py "carpeta/documento.pdf" ["otro.pdf" ...]
"""
import sys

from pipeline.cli import main

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python convertir_pymupdf.py archivo1.pdf [archivo2.pdf ...]", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(["convertir", "--engine", "pymupdf", *sys.argv[1:]]))
