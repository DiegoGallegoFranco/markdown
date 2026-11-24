"""Generadores de archivos sintéticos para los tests (imágenes, PDFs, pptx).

Todo se crea con pymupdf en tmp_path: la suite no depende de documentos reales.
"""
from __future__ import annotations

from pathlib import Path

import pymupdf

LOREM = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod "
         "tempor incididunt ut labore et dolore magna aliqua. ") * 4


def png(path: Path, w: int, h: int, valor: int = 128, alpha: bool = False) -> Path:
    """PNG de color liso. `valor` distinto -> bytes distintos (hash distinto)."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, w, h), alpha)
    pix.clear_with(valor)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pix.tobytes("png"))
    return path


def jpg(path: Path, w: int, h: int, valor: int = 128) -> Path:
    """JPEG de color liso, con la extensión que diga `path` (puede mentir)."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, w, h), False)
    pix.clear_with(valor)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pix.tobytes("jpg"))
    return path


# --------------------------------------------------------------------------- PDFs

def pagina_nativa(doc: pymupdf.Document, i: int = 0) -> None:
    p = doc.new_page()
    p.insert_textbox(pymupdf.Rect(50, 50, 545, 790), f"Página {i}. " + LOREM, fontsize=11)


_RASTER: bytes | None = None


def _raster_de_pagina() -> bytes:
    """PNG de una página de texto rasterizada (lo que produce un escáner)."""
    global _RASTER
    if _RASTER is None:
        tmp = pymupdf.open()
        pagina_nativa(tmp)
        _RASTER = tmp[0].get_pixmap(dpi=40).tobytes("png")
        tmp.close()
    return _RASTER


def pagina_escaneada(doc: pymupdf.Document, cobertura: float = 1.0) -> None:
    """Página con SOLO una imagen rasterizada (sin capa de texto).

    `cobertura` < 1 encoge la imagen (fracción del área de la página).
    """
    p = doc.new_page()
    r = p.rect
    if cobertura < 1.0:
        f = cobertura ** 0.5
        r = pymupdf.Rect(0, 0, r.width * f, r.height * f)
    p.insert_image(r, stream=_raster_de_pagina(), keep_proportion=False)


def pagina_blanca(doc: pymupdf.Document) -> None:
    doc.new_page()


def pdf(path: Path, paginas: str) -> Path:
    """Crea un PDF a partir de una receta: 'n' nativa, 'e' escaneada, 'b' blanca.

    Ej.: pdf(p, 'e' + 'n' * 29) -> portada escaneada + 29 páginas de texto.
    """
    doc = pymupdf.open()
    for i, c in enumerate(paginas):
        {"n": lambda: pagina_nativa(doc, i),
         "e": lambda: pagina_escaneada(doc),
         "b": lambda: pagina_blanca(doc)}[c]()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path
