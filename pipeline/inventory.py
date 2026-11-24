"""Fase 1 — inventario y diagnóstico de PDFs.

Clasifica cada PDF como `nativo` (texto seleccionable), `escaneado` (necesita
OCR real) o `mixto` (mayormente nativo, pero con páginas escaneadas dentro).
Esa clasificación es la que decide el motor de conversión: los escaneados y los
mixtos solo los resuelve Marker; los nativos simples salen bien y muchísimo más
rápido con pymupdf4llm.

Por qué existe `mixto`: pymupdf4llm NO hace OCR. Un PDF con 70% de páginas
nativas y 30% escaneadas (anexos, firmas, planos) pasaba el umbral de "nativo",
iba entero a pymupdf4llm y esas páginas salían vacías en el Markdown, sin aviso.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import fitz  # PyMuPDF

FIELDNAMES = [
    "ruta", "paginas", "tipo", "paginas_con_texto", "ratio_texto",
    "imagenes_embebidas", "tam_mb", "motor_sugerido",
    # Columnas añadidas al final (no romper a quien lea el CSV por posición):
    # explican POR QUÉ se eligió el motor, sobre todo cuando un PDF que parece
    # nativo acaba en Marker.
    "paginas_escaneadas", "paginas_en_blanco", "lista_escaneadas", "motivo",
]


def _env_float(name: str, default: float) -> float:
    # Un valor mal escrito en .env no debe tumbar el inventario entero: se
    # ignora y se usa el valor por defecto.
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Un PDF con menos de este ratio de páginas con texto (sobre las páginas con
# contenido, ver abajo) se considera escaneado.
TEXT_RATIO_NATIVO = 0.6
# Caracteres mínimos para considerar que una página "tiene texto" (evita contar
# como nativa una página escaneada con solo un número de página en OCR previo).
MIN_CHARS_PAGINA = 30

# Fracción del área de la página que deben cubrir las imágenes para que una
# página SIN texto se considere escaneada. Un escaneo es una imagen a página
# completa (~100%, a veces troceada en tiras, por eso se suman las áreas); 0.5
# deja margen para escaneos recortados o con bordes, y excluye páginas sin
# texto que solo llevan un logo o una foto pequeña.
COBERTURA_ESCANEO = _env_float("DOC2MD_COBERTURA_ESCANEO", 0.5)

# Una página sin texto y sin imagen grande cuyo content stream ocupa menos que
# esto se considera en blanco (separador, contraportada, "página par" vacía).
# No cuenta para nada: no es escaneada ni debe diluir el ratio de texto. Si el
# stream es mayor (dibujo vectorial, plano con texto convertido a trazos) se
# conserva el comportamiento original: cuenta como página sin texto.
MAX_BYTES_PAGINA_BLANCA = 1024

# Cuántas páginas escaneadas hacen falta para mandar a Marker un PDF que por lo
# demás es nativo. Se exige AL MENOS UNA de estas dos condiciones:
#   - DOC2MD_MIXTO_MIN_PAGINAS (2) páginas escaneadas en términos absolutos, o
#   - DOC2MD_MIXTO_MIN_RATIO (10%) de las páginas con contenido.
# Por qué: una sola página-imagen en un documento largo (portada fotográfica,
# un diagrama a página completa) es casi siempre decorativa, y mandar 300
# páginas a Marker por ella cuesta horas (Marker escala con el nº de páginas).
# Con 2+ páginas escaneadas ya hay un patrón (anexo escaneado, actas firmadas)
# y perderlas en silencio es peor que la lentitud, que al menos es visible. En
# documentos cortos una sola página escaneada sí basta (pasa el 10%): ahí es
# una parte grande del contenido y el sobrecoste de Marker es pequeño.
# Para ser más agresivo (cero pérdidas) basta DOC2MD_MIXTO_MIN_PAGINAS=1.
MIXTO_MIN_PAGINAS = _env_int("DOC2MD_MIXTO_MIN_PAGINAS", 2)
MIXTO_MIN_RATIO = _env_float("DOC2MD_MIXTO_MIN_RATIO", 0.10)

# Límite de páginas listadas en el CSV, para que una celda no crezca sin fin
# en documentos de cientos de páginas escaneadas.
MAX_LISTA_ESCANEADAS = 20


def _image_coverage(page: fitz.Page) -> float:
    """Fracción del área de la página cubierta por imágenes (0..1).

    Usa get_image_info() (bboxes de lo que realmente se dibuja, incluidas
    imágenes inline y dentro de Form XObjects) en vez de renderizar: es
    órdenes de magnitud más barato. Las áreas se suman recortadas a la página
    y se acotan a 1; los solapes son raros en escaneos y sobrestimar un poco
    no cambia la decisión.
    """
    page_rect = page.rect
    page_area = abs(page_rect)
    if not page_area:
        return 0.0
    covered = 0.0
    for info in page.get_image_info():
        r = fitz.Rect(info["bbox"]) & page_rect
        if not r.is_empty:
            covered += abs(r)
    return min(covered / page_area, 1.0)


def _is_blank(page: fitz.Page) -> bool:
    # Solo se llama en páginas sin texto ni imagen grande (raras), así que
    # leer el content stream no afecta al rendimiento del caso común.
    try:
        return len(page.read_contents()) < MAX_BYTES_PAGINA_BLANCA
    except Exception:
        return False


def analyze_pdf(path: Path, root: Path) -> dict:
    doc = fitz.open(path)
    try:
        n_pages = len(doc)
        pages_with_text = 0
        scanned_pages: list[int] = []  # 1-based, para el humano que lee el CSV
        blank_pages = 0
        embedded_images = set()

        # Una sola pasada por página. Lo caro adicional (get_image_info,
        # read_contents) solo se ejecuta en páginas SIN texto, que en un
        # corpus mayormente nativo son pocas.
        for page in doc:
            if len(page.get_text("text").strip()) > MIN_CHARS_PAGINA:
                pages_with_text += 1
            elif _image_coverage(page) >= COBERTURA_ESCANEO:
                scanned_pages.append(page.number + 1)
            elif _is_blank(page):
                blank_pages += 1
            # get_images() es caro; una sola llamada por página (el código
            # original la hacía dos veces y descartaba la primera).
            for img in page.get_images(full=True):
                embedded_images.add(img[0])  # xref: dedup de imágenes compartidas
    finally:
        doc.close()

    n_scanned = len(scanned_pages)
    # ratio_texto se reporta como siempre (sobre el total de páginas) para no
    # cambiar el significado de la columna; la DECISIÓN, en cambio, excluye
    # las páginas en blanco: un nativo con muchos separadores vacíos no debe
    # acabar clasificado como escaneado.
    text_ratio = pages_with_text / n_pages if n_pages else 0
    content_pages = n_pages - blank_pages
    decision_ratio = pages_with_text / content_pages if content_pages else 0

    if decision_ratio < TEXT_RATIO_NATIVO:
        tipo = "escaneado"
        motivo = f"solo {pages_with_text}/{content_pages} páginas con contenido tienen texto"
    elif n_scanned and (
        n_scanned >= MIXTO_MIN_PAGINAS
        or n_scanned / content_pages >= MIXTO_MIN_RATIO
    ):
        tipo = "mixto"
        motivo = (f"{n_scanned} página(s) sin texto cubierta(s) por imagen; "
                  "pymupdf4llm las dejaría vacías (no hace OCR)")
    else:
        tipo = "nativo"
        motivo = ""
        if n_scanned:
            # Por debajo del umbral: se avisa en el CSV para que el usuario
            # pueda forzar Marker si esa página sí importaba.
            motivo = (f"{n_scanned} página(s)-imagen ignorada(s) por umbral "
                      f"(min {MIXTO_MIN_PAGINAS} o {MIXTO_MIN_RATIO:.0%})")

    lista = ",".join(str(p) for p in scanned_pages[:MAX_LISTA_ESCANEADAS])
    if n_scanned > MAX_LISTA_ESCANEADAS:
        lista += ",…"

    return {
        "ruta": str(path.resolve().relative_to(root)) if _under(path, root) else str(path),
        "paginas": n_pages,
        "tipo": tipo,
        "paginas_con_texto": pages_with_text,
        "ratio_texto": round(text_ratio, 2),
        "imagenes_embebidas": len(embedded_images),
        "tam_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "motor_sugerido": "pymupdf" if tipo == "nativo" else "marker",
        "paginas_escaneadas": n_scanned,
        "paginas_en_blanco": blank_pages,
        "lista_escaneadas": lista,
        "motivo": motivo,
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
            row["motivo"] = f"no se pudo analizar: {e}"
            row["tam_mb"] = round(Path(pdf).stat().st_size / (1024 * 1024), 2)
            if on_progress:
                on_progress(f"ERROR analizando {pdf}: {e}")
        rows.append(row)
        if on_progress:
            extra = f" ({row['motivo']})" if row.get("tipo") == "mixto" else ""
            on_progress(f"{row['ruta']}: {row['paginas']} pag, {row['tipo']}{extra}")
    return rows


def write_csv(rows: list[dict], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return out_path
