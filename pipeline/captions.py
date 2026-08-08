"""Captioning de imágenes con la API de Claude (visión), en paralelo.

Por qué: un RAG de solo texto no puede "ver" una imagen. Generar una vez una
descripción corta de cada diagrama/mapa/tabla-como-imagen y dejarla como texto
justo debajo de la imagen hace ese contenido buscable.

Por qué archivos separados y no base64: el costo en tokens de un modelo de
visión depende de la RESOLUCIÓN de la imagen, no del encoding. Base64 no ahorra
nada, infla el .md ~33% y ensucia los diffs de git.

Es REANUDABLE: antes de llamar a la API se detecta qué imágenes ya tienen
caption y se saltan. Volver a ejecutarlo tras un fallo a mitad de lote no
repaga la API ni duplica descripciones.

Requiere ANTHROPIC_API_KEY en el entorno o en un .env.
"""
from __future__ import annotations

import base64
import concurrent.futures
import os
import threading
import time
from pathlib import Path

from .mdimages import image_refs, ref_name, reference_pattern

# Formatos que acepta la API de visión de Claude.
IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
MEDIA_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
}

MARKER = "*[descripción generada por IA:"

# Sonnet es el punto dulce calidad/costo para describir una imagen en 1-2 frases;
# con Haiku el lote sale ~3x más barato y para logos/fotos genéricas basta.
DEFAULT_MODEL = os.environ.get("CAPTION_MODEL", "claude-sonnet-5")
DEFAULT_WORKERS = int(os.environ.get("CAPTION_WORKERS", "8"))
# El SDK ya reintenta 429/5xx con backoff exponencial; se sube el tope por
# defecto (2) porque un lote de cientos de imágenes en paralelo los provoca.
DEFAULT_MAX_RETRIES = int(os.environ.get("CAPTION_MAX_RETRIES", "6"))

PROMPT = (
    "Describe brevemente el contenido de esta imagen en español, en 1-2 "
    "frases. Es una imagen extraída de un documento (informe, manual o "
    "artículo). Si es un diagrama, mapa o esquema técnico, describe qué "
    "representa (elementos clave, relaciones, términos visibles). Si es "
    "solo un logo, decoración o foto genérica sin contenido informativo "
    "relevante, dilo brevemente. No agregues preámbulo, responde solo con "
    "la descripción."
)


# --------------------------------------------------------------------------- estado de captions

def caption_state(md_path: Path) -> dict[str, bool]:
    """{nombre_de_imagen: ya_tiene_caption} para la PRIMERA aparición de cada una.

    Se mira solo la primera aparición porque es la que se anota: si Marker
    duplicó una página, la segunda referencia es contenido redundante, no
    información perdida.
    """
    try:
        text = md_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    paras = text.split("\n\n")
    estado: dict[str, bool] = {}
    for i, para in enumerate(paras):
        # Un párrafo puede traer varias imágenes; se miran todas.
        for ref in image_refs(para):
            name = ref_name(ref)
            if name in estado:
                continue  # solo la primera aparición
            estado[name] = i + 1 < len(paras) and paras[i + 1].lstrip().startswith(MARKER)
    return estado


def md_files_for(img_path: Path) -> list[Path]:
    """Los .md que pueden referenciar esta imagen (misma carpeta o su padre)."""
    doc_dir = img_path.parent
    if doc_dir.name in ("assets", "media"):
        doc_dir = doc_dir.parent
    return sorted(doc_dir.glob("*.md"))


def pending_images(output_dir: Path, limit=None) -> tuple[list[Path], int]:
    """Imágenes que aún no tienen caption. Devuelve (pendientes, total_encontradas)."""
    output_dir = Path(output_dir)
    todas = sorted(p for p in output_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    cache: dict[Path, dict[str, bool]] = {}
    pendientes = []
    for img in todas:
        for md in md_files_for(img):
            estado = cache.get(md)
            if estado is None:
                estado = cache[md] = caption_state(md)
            if estado.get(img.name) is False:  # referenciada y sin caption
                pendientes.append(img)
                break
    if limit:
        pendientes = pendientes[:limit]
    return pendientes, len(todas)


# --------------------------------------------------------------------------- escritura

_md_locks: dict[Path, threading.Lock] = {}
_md_locks_guard = threading.Lock()


def _lock_for(md_path: Path) -> threading.Lock:
    # Varias imágenes del mismo documento se procesan en paralelo y todas hacen
    # read-modify-write del MISMO .md. Sin lock por archivo, dos hilos leen la
    # misma versión y el segundo write pisa al primero (lost update) -> captions
    # que desaparecen sin error visible. No quitar.
    with _md_locks_guard:
        return _md_locks.setdefault(md_path, threading.Lock())


def insert_caption(md_path: Path, img_name: str, caption: str) -> bool:
    """Inserta el caption bajo la primera aparición de la imagen. Idempotente."""
    pattern = reference_pattern(img_name)
    marca = f"\n\n*[descripción generada por IA: {caption}]*"
    with _lock_for(md_path):
        # Se relee dentro del lock: otro hilo pudo haber escrito ya.
        if caption_state(md_path).get(img_name, True):
            return False
        text = md_path.read_text(encoding="utf-8")
        # m.group(0): la etiqueta completa, sea ![](x) o <img ...> — esta última
        # puede ocupar varias líneas, así que hay que conservarla entera.
        new_text, n = pattern.subn(lambda m: m.group(0) + marca, text, count=1)
        if n:
            md_path.write_text(new_text, encoding="utf-8")
        return n > 0


# --------------------------------------------------------------------------- API

def caption_image(client, img_path: Path, model=DEFAULT_MODEL) -> str:
    data = base64.standard_b64encode(img_path.read_bytes()).decode("utf-8")
    media_type = MEDIA_TYPES[img_path.suffix.lower()]
    resp = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    )
    if resp.stop_reason == "refusal":
        return "[el modelo declinó describir esta imagen]"
    return next((b.text.strip() for b in resp.content if b.type == "text"), "")


def run(output_dir, model=DEFAULT_MODEL, workers=DEFAULT_WORKERS, limit=None,
        images=None, log=None, dry_run=False) -> dict:
    """Genera e inserta captions. Devuelve un resumen con contadores."""
    import anthropic

    def _log(msg):
        if log:
            log(msg)

    output_dir = Path(output_dir)
    if images is None:
        images, total = pending_images(output_dir, limit=limit)
    else:
        images = [Path(p) for p in images]
        total = len(images)

    _log(f"Imágenes sin caption: {len(images)} (de {total} encontradas)")
    resumen = {"pendientes": len(images), "total": total, "ok": 0, "errores": 0,
               "sin_referencia": 0, "duracion_s": 0.0, "modelo": model}
    if dry_run or not images:
        return resumen

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("falta ANTHROPIC_API_KEY (variable de entorno o .env)")

    client = anthropic.Anthropic(max_retries=DEFAULT_MAX_RETRIES)
    contador = threading.Lock()
    hechas = 0

    def process(img_path: Path):
        nonlocal hechas
        try:
            caption = caption_image(client, img_path, model=model)
            inserted = any(insert_caption(md, img_path.name, caption) for md in md_files_for(img_path))
            with contador:
                hechas += 1
                if inserted:
                    resumen["ok"] += 1
                else:
                    resumen["sin_referencia"] += 1
                _log(f"[{hechas}/{len(images)}] {'OK' if inserted else 'sin referencia en .md'}: "
                     f"{img_path.name} -> {caption[:80]}")
        except anthropic.RateLimitError as e:
            with contador:
                hechas += 1
                resumen["errores"] += 1
                _log(f"[ERROR rate-limit] {img_path.name}: {e} "
                     f"(reintentar el lote: los ya hechos se saltan solos)")
        except Exception as e:
            with contador:
                hechas += 1
                resumen["errores"] += 1
                _log(f"[ERROR] {img_path.name}: {e}")

    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(process, images))
    resumen["duracion_s"] = round(time.time() - started, 1)

    _log(f"Captions completados en {resumen['duracion_s']/60:.1f} min. "
         f"OK: {resumen['ok']} | sin referencia: {resumen['sin_referencia']} | errores: {resumen['errores']}")
    return resumen


# --------------------------------------------------------------------------- auditoría / limpieza

def audit(output_dir) -> list[str]:
    """Imágenes referenciadas y existentes que se quedaron sin caption."""
    pendientes, _ = pending_images(Path(output_dir))
    return [str(p) for p in pendientes]


def dedup(output_dir, log=None) -> dict:
    """Elimina captions duplicados consecutivos, dejando el primero."""
    total_removed = files_changed = 0
    for md in sorted(Path(output_dir).rglob("*.md")):
        text = md.read_text(encoding="utf-8", errors="replace")
        if MARKER not in text:
            continue
        paras = text.split("\n\n")
        out, removed, i = [], 0, 0
        while i < len(paras):
            out.append(paras[i])
            if paras[i].lstrip().startswith(MARKER):
                j = i + 1
                while j < len(paras) and paras[j].lstrip().startswith(MARKER):
                    removed += 1
                    j += 1
                i = j
            else:
                i += 1
        if removed:
            md.write_text("\n\n".join(out), encoding="utf-8")
            files_changed += 1
            total_removed += removed
            if log:
                log(f"{md}: -{removed} duplicados")
    return {"duplicados_eliminados": total_removed, "archivos_modificados": files_changed}
