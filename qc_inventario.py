#!/usr/bin/env python3
"""Fase 4 — control de calidad e inventario final.

Wrapper de `python -m pipeline qc`.

Uso:
    python qc_inventario.py [directorio_output]
"""
import sys

from pipeline.cli import main

if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "output"
    sys.exit(main(["qc", "--output", output]))
