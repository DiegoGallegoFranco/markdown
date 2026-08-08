#!/usr/bin/env python3
"""Fase 1 — inventario de PDFs. Wrapper de `python -m pipeline inventario`.

Uso:
    python inventario.py [directorio]   # por defecto, el directorio actual
"""
import sys

from pipeline.cli import main

if __name__ == "__main__":
    ruta = sys.argv[1] if len(sys.argv) > 1 else "."
    sys.exit(main(["inventario", ruta, *sys.argv[2:]]))
