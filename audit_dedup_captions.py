#!/usr/bin/env python3
"""Audita y limpia los captions generados.

Wrapper de `python -m pipeline captions-audit` / `captions-dedup`.

Uso:
    python audit_dedup_captions.py --audit
    python audit_dedup_captions.py --dedup
"""
import argparse
import sys

from pipeline.cli import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audit", action="store_true")
    group.add_argument("--dedup", action="store_true")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()

    sub = "captions-audit" if args.audit else "captions-dedup"
    sys.exit(main([sub, "--output", args.output]))
