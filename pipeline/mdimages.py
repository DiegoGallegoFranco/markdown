"""Referencias a imágenes en Markdown, en los dos formatos que produce el pipeline.

pymupdf4llm, Marker y el conversor de pptx emiten `![alt](ruta)`. **pandoc no**:
cuando la imagen del .docx trae atributos (tamaño, alineación) emite HTML,
`<img src="ruta" style="..." />`, y además parte la etiqueta en varias líneas.

Si solo se busca la forma Markdown, las imágenes de los docx quedan invisibles:
el control de calidad las cuenta como 0 y el captioning nunca las describe.
Este módulo es la única definición de "referencia a imagen" del proyecto.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlparse

# ![alt](ruta "titulo opcional")
MD_IMG_RE = re.compile(r'!\[[^\]]*\]\(\s*<?([^)\s>]+)>?[^)]*\)')
# <img ... src="ruta" ... > — DOTALL implícito: la etiqueta puede ocupar varias
# líneas, así que no se puede asumir que src esté en la misma línea que <img.
HTML_IMG_RE = re.compile(r'<img\b[^>]*?\bsrc\s*=\s*["\']([^"\']+)["\'][^>]*>',
                         re.IGNORECASE | re.DOTALL)
# Cualquiera de las dos, para partir un documento por sus imágenes.
ANY_IMG_RE = re.compile(
    r'(?:!\[[^\]]*\]\(\s*<?[^)\s>]+>?[^)]*\))'
    r'|(?:<img\b[^>]*?\bsrc\s*=\s*["\'][^"\']+["\'][^>]*>)',
    re.IGNORECASE | re.DOTALL,
)


def is_external(ref: str) -> bool:
    return urlparse(ref).scheme in ("http", "https", "data")


def clean_ref(ref: str) -> str:
    """Normaliza una ruta de referencia: sin ancla, sin query, sin percent-encoding."""
    return unquote(ref.split("#")[0].split("?")[0]).strip()


def image_refs(text: str) -> list[str]:
    """Todas las rutas de imagen referenciadas, en ambos formatos."""
    return MD_IMG_RE.findall(text) + HTML_IMG_RE.findall(text)


def local_image_refs(text: str) -> list[str]:
    return [r for r in image_refs(text) if not is_external(r)]


def resolve(md_path: Path, ref: str) -> Path:
    return md_path.parent / clean_ref(ref)


def ref_name(ref: str) -> str:
    """Nombre de archivo de una referencia (para casar imagen <-> referencia)."""
    return Path(clean_ref(ref)).name


def reference_pattern(img_name: str) -> re.Pattern:
    """Casa una referencia concreta a `img_name`, en cualquiera de los dos formatos.

    Se usa para insertar el caption justo detrás de la etiqueta, así que la
    captura tiene que abarcar la etiqueta COMPLETA (incluido el `/>` de cierre
    del HTML, que puede estar líneas más abajo).
    """
    esc = re.escape(img_name)
    return re.compile(
        r'(!\[[^\]]*\]\([^)]*' + esc + r'[^)]*\))'
        r'|(<img\b[^>]*?\bsrc\s*=\s*["\'][^"\']*' + esc + r'["\'][^>]*>)',
        re.IGNORECASE | re.DOTALL,
    )
