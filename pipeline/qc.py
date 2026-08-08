"""Fase 4 — control de calidad e inventario final.

Recorre la salida, valida cada .md (vacío, sin encabezados, enlaces de imagen
rotos) y produce un inventario por documento. Un .md con 0 encabezados en un
documento largo, o con imágenes referenciadas que no existen, es señal de
revisión manual — no necesariamente un error del pipeline (un docx que no usa
estilos de encabezado sale así legítimamente).
"""
from __future__ import annotations

import csv
from pathlib import Path

from .mdimages import local_image_refs, resolve

CAPTION_MARKER = "*[descripción generada por IA:"

FIELDNAMES = [
    "ruta_md", "duracion_min", "chars", "encabezados", "imagenes",
    "captions", "enlaces_rotos", "estado",
]


def check_md(md: Path, root: Path, duracion_s=None) -> dict:
    text = md.read_text(encoding="utf-8", errors="replace")
    n_chars = len(text.strip())
    n_headers = sum(1 for line in text.splitlines() if line.strip().startswith("#"))
    # local_image_refs cubre ![](x) y <img src="x">: pandoc emite la segunda
    # forma para imágenes con atributos, y sin ella los docx daban 0 imágenes.
    refs = local_image_refs(text)
    broken = [r for r in refs if not resolve(md, r).exists()]
    n_captions = text.count(CAPTION_MARKER)

    if n_chars == 0:
        estado = "ERROR: vacío"
    elif broken:
        estado = f"REVISION MANUAL: {len(broken)} enlace(s) roto(s)"
    elif n_headers == 0:
        estado = "REVISION MANUAL: sin encabezados"
    else:
        estado = "OK"

    try:
        ruta = str(md.resolve().relative_to(Path(root).resolve()))
    except ValueError:
        ruta = str(md)

    return {
        "ruta_md": ruta,
        # `is not None`, no truthiness: una conversión de 0.0 s (un pptx
        # pequeño) es un dato válido, no un dato ausente.
        "duracion_min": round(duracion_s / 60, 1) if duracion_s is not None else "",
        "chars": n_chars,
        "encabezados": n_headers,
        "imagenes": len(refs),
        "captions": n_captions,
        "enlaces_rotos": len(broken),
        "estado": estado,
    }


def run(output_dir, root=None, durations=None) -> list[dict]:
    """durations: {stem_del_documento: segundos}, opcional."""
    output_dir = Path(output_dir)
    root = Path(root) if root else output_dir.parent
    durations = durations or {}
    return [
        check_md(md, root, durations.get(md.stem))
        for md in sorted(output_dir.rglob("*.md"))
    ]


def write_csv(rows: list[dict], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def summarize(rows: list[dict]) -> dict:
    return {
        "documentos": len(rows),
        "ok": sum(1 for r in rows if r["estado"] == "OK"),
        "revision": sum(1 for r in rows if r["estado"].startswith("REVISION")),
        "error": sum(1 for r in rows if r["estado"].startswith("ERROR")),
        "imagenes": sum(r["imagenes"] for r in rows),
        "captions": sum(r["captions"] for r in rows),
    }
