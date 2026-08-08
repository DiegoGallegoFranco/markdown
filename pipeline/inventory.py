"""Fase 1 — inventario y diagnóstico de PDFs.

Clasifica cada PDF como `nativo` (texto seleccionable) o `escaneado` (necesita
OCR real). Esa clasificación es la que decide el motor de conversión: los
escaneados solo los resuelve Marker; los nativos simples salen bien y muchísimo
más rápido con pymupdf4llm.
"""
from __future__ import annotations

import csv
from pathlib import Path

import fitz  # PyMuPDF

FIELDNAMES = [
    "ruta", "paginas", "tipo", "paginas_con_texto", "ratio_texto",
    "imagenes_embebidas", "tam_mb", "motor_sugerido",
]

# Un PDF con menos de este ratio de páginas con texto se considera escaneado.
TEXT_RATIO_NATIVO = 0.6
# Caracteres mínimos para considerar que una página "tiene texto" (evita contar
# como nativa una página escaneada con solo un número de página en OCR previo).
MIN_CHARS_PAGINA = 30


def analyze_pdf(path: Path, root: Path) -> dict:
    doc = fitz.open(path)
    try:
        n_pages = len(doc)
        pages_with_text = 0
        embedded_images = set()

        for page in doc:
            if len(page.get_text("text").strip()) > MIN_CHARS_PAGINA:
                pages_with_text += 1
            # get_images() es caro; una sola llamada por página (el código
            # original la hacía dos veces y descartaba la primera).
            for img in page.get_images(full=True):
                embedded_images.add(img[0])  # xref: dedup de imágenes compartidas
    finally:
        doc.close()

    text_ratio = pages_with_text / n_pages if n_pages else 0
    tipo = "nativo" if text_ratio >= TEXT_RATIO_NATIVO else "escaneado"

    return {
        "ruta": str(path.resolve().relative_to(root)) if _under(path, root) else str(path),
        "paginas": n_pages,
        "tipo": tipo,
        "paginas_con_texto": pages_with_text,
        "ratio_texto": round(text_ratio, 2),
        "imagenes_embebidas": len(embedded_images),
        "tam_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "motor_sugerido": "marker" if tipo == "escaneado" else "pymupdf",
    }


def _under(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def build_inventory(pdfs, root: Path, on_progress=None) -> list[dict]:
    rows = []
    for pdf in pdfs:
        try:
            row = analyze_pdf(Path(pdf), Path(root))
        except Exception as e:
            row = {k: "" for k in FIELDNAMES}
            row["ruta"] = str(pdf)
            row["tipo"] = "ERROR"
            row["paginas"] = "ERROR"
            row["motor_sugerido"] = "marker"  # ante la duda, el motor robusto
            row["tam_mb"] = round(Path(pdf).stat().st_size / (1024 * 1024), 2)
            if on_progress:
                on_progress(f"ERROR analizando {pdf}: {e}")
        rows.append(row)
        if on_progress:
            on_progress(f"{row['ruta']}: {row['paginas']} pag, {row['tipo']}")
    return rows


def write_csv(rows: list[dict], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return out_path
