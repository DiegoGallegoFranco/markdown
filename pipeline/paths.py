"""Resolución de rutas de entrada/salida, compartida por todos los conversores.

El problema que resuelve: los conversores originales hacían
`path.resolve().relative_to(Path.cwd())`, que lanza ValueError en cuanto el
documento vive fuera del directorio actual — que es justo el flujo documentado
(la carpeta de documentos NO vive en el repo). Aquí el "root" de entrada es
explícito y se deriva del propio lote cuando no se indica.
"""
from __future__ import annotations

import os
from pathlib import Path

# Extensiones soportadas por cada conversor.
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}
PPTX_EXTS = {".pptx"}
SUPPORTED_EXTS = PDF_EXTS | DOCX_EXTS | PPTX_EXTS


def slugify(name: str) -> str:
    """Sanea un componente de ruta.

    pymupdf4llm sanea internamente la ruta que usa para `pix.save()` pero solo
    después de hacer mkdir con la ruta sin sanear; si el nombre lleva espacios o
    paréntesis ambas rutas divergen y el guardado de imágenes falla. Marker tiene
    un problema parecido al nombrar su subcarpeta de salida. Se sanea antes de
    pasar cualquier ruta a esas librerías.
    """
    return name.replace("(", "-").replace(")", "-").replace(" ", "_")


def common_root(paths) -> Path:
    """Directorio común de un lote de archivos, para acortar las rutas de salida.

    Con un solo archivo es su carpeta; con varios, el ancestro común. Si no hay
    archivos, el directorio actual.
    """
    resolved = [Path(p).resolve() for p in paths]
    if not resolved:
        return Path.cwd()
    if len(resolved) == 1:
        return resolved[0].parent
    return Path(os.path.commonpath([str(p) for p in resolved]))


def relative_key(path, root) -> Path:
    """Ruta de `path` relativa a `root`, sin lanzar si queda fuera.

    Si el archivo no está bajo `root` (ruta absoluta ajena, otro volumen), se
    espeja la ruta absoluta sin el ancla: `/Users/x/doc.pdf` -> `Users/x/doc.pdf`.
    Es feo pero determinista y libre de colisiones, y nunca genera rutas como
    `output//Users/...` (que es lo que producían los scripts de bash).
    """
    p = Path(path).resolve()
    r = Path(root).resolve()
    try:
        return p.relative_to(r)
    except ValueError:
        return Path(*p.parts[1:]) if p.is_absolute() else Path(p)


def doc_out_dir(input_path, out_root, input_root) -> Path:
    """Carpeta de salida autocontenida para un documento.

    `<out_root>/<estructura relativa saneada>/<nombre-del-documento-saneado>/`
    """
    rel = relative_key(input_path, input_root)
    parts = [slugify(part) for part in rel.parent.parts]
    stem = slugify(Path(input_path).stem)
    return Path(out_root).joinpath(*parts, stem)


def collect_documents(target, exts=SUPPORTED_EXTS) -> list[Path]:
    """Expande un archivo o directorio a la lista de documentos soportados."""
    target = Path(target)
    if target.is_file():
        return [target] if target.suffix.lower() in exts else []
    return sorted(
        p for p in target.rglob("*")
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("~$")
    )
