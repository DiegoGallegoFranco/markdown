"""Captioning de imágenes con un modelo de visión, en paralelo.

Por qué: un RAG de solo texto no puede "ver" una imagen. Generar una vez una
descripción corta de cada diagrama/mapa/tabla-como-imagen y dejarla como texto
justo debajo de la imagen hace ese contenido buscable.

Por qué archivos separados y no base64: el costo en tokens de un modelo de
visión depende de la RESOLUCIÓN de la imagen, no del encoding. Base64 no ahorra
nada, infla el .md ~33% y ensucia los diffs de git.

Es REANUDABLE: antes de llamar al modelo se detecta qué imágenes ya tienen
caption y se saltan. Volver a ejecutarlo tras un fallo a mitad de lote no
repite trabajo ni duplica descripciones.

Evita llamadas inútiles al modelo: las imágenes con bytes idénticos (el logo
de cada página) se describen una sola vez, las diminutas (iconos, viñetas,
filetes) se omiten, y las grandes se reducen en memoria antes de enviarlas.

Dos backends, misma interfaz (CAPTION_BACKEND):

    ollama     (default) modelo local vía Ollama. Las imágenes NO salen de la
               máquina. Sin clave, sin costo, más lento.
    anthropic  API de Claude. Más rápido y mejor calidad, pero cada imagen
               viaja a un tercero — no usar con documentos sensibles.
"""
from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .mdimages import image_refs, ref_name, reference_pattern

# Formatos que aceptan ambos backends.
IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
MEDIA_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
}

MARKER = "*[descripción generada por IA:"

BACKEND_OLLAMA, BACKEND_ANTHROPIC = "ollama", "anthropic"
# Local por defecto: es la opción que no filtra documentos. Cambiar a
# "anthropic" es una decisión consciente, no algo en lo que caer por descuido.
CAPTION_BACKEND = os.environ.get("CAPTION_BACKEND", BACKEND_OLLAMA).strip().lower()

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
# Un modelo de 30B describiendo una imagen tarda decenas de segundos; urllib sin
# timeout se cuelga para siempre si el servidor no responde.
OLLAMA_TIMEOUT_S = int(os.environ.get("OLLAMA_TIMEOUT_S", "300"))

DEFAULT_MODELS = {
    # qwen3-vl es el más sólido en documentos: OCR robusto con texto borroso o
    # inclinado, y buena lectura de diagramas y esquemas técnicos.
    BACKEND_OLLAMA: "qwen3-vl:32b",
    BACKEND_ANTHROPIC: "claude-sonnet-5",
}
DEFAULT_MODEL = os.environ.get("CAPTION_MODEL") or DEFAULT_MODELS.get(
    CAPTION_BACKEND, DEFAULT_MODELS[BACKEND_OLLAMA])

# Ollama serializa las peticiones por defecto: más hilos contra un modelo de
# 30B no aceleran nada y pueden agotar la VRAM (mismo valor que docker-compose).
# La API de Claude sí paraleliza.
DEFAULT_WORKERS = int(os.environ.get(
    "CAPTION_WORKERS", "1" if CAPTION_BACKEND == BACKEND_OLLAMA else "8"))
DEFAULT_MAX_RETRIES = int(os.environ.get("CAPTION_MAX_RETRIES", "6"))

# Umbral de "imagen diminuta", sobre el LADO MENOR en píxeles. Se usa el lado
# menor y no el área porque así caen tanto los iconos/viñetas (16x16, 32x32)
# como las líneas separadoras y filetes (800x3), que tienen área grande pero
# ningún contenido que describir. 0 desactiva el filtro.
CAPTION_MIN_PX = int(os.environ.get("CAPTION_MIN_PX", "64"))
# Lado mayor máximo de lo que se ENVÍA al modelo. El coste y el tiempo de un VLM
# crecen con la resolución, y para una descripción de 1-2 frases 1024 px sobra.
# Se reduce solo en memoria: el archivo en disco es el que enlaza el .md y no
# se toca. 0 desactiva la reducción.
CAPTION_MAX_PX = int(os.environ.get("CAPTION_MAX_PX", "1024"))

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


# --------------------------------------------------------------------------- imágenes: tamaño y preparación

# Decisión sobre las imágenes DIMINUTAS (lado menor < CAPTION_MIN_PX):
# no se describen y tampoco se escribe nada en el .md; simplemente NO cuentan
# como pendientes (ni aquí, ni en audit(), ni en `captions-audit`).
#   - No se describen: un icono o una viñeta no aporta nada buscable y cada
#     llamada a un VLM local cuesta decenas de segundos.
#   - No se marca en el .md (p. ej. "[imagen decorativa]"): ese texto acabaría
#     troceado dentro de los chunks del RAG como ruido, repetido en cada página.
#   - Se excluyen por definición en vez de persistir un estado: la condición se
#     recalcula cada vez desde el propio archivo, así que si se baja el umbral
#     reaparecen como pendientes sin migrar nada, y la auditoría no se llena de
#     "faltantes" que nunca se van a resolver.
# Si el tamaño no se puede leer (formato que pymupdf no decodifica, p. ej.
# webp), la imagen NO se considera diminuta: ante la duda, se describe.

def _pymupdf():
    # Import perezoso: este módulo lo importa la web solo para consultar el
    # estado del backend, y no necesita cargar MuPDF para eso.
    import pymupdf
    return pymupdf


def image_size(img_path: Path) -> tuple[int, int] | None:
    """(ancho, alto) en píxeles, o None si pymupdf no sabe decodificarla."""
    try:
        pix = _pymupdf().Pixmap(str(img_path))
        return pix.width, pix.height
    except Exception:
        return None


def is_tiny(img_path: Path, min_px=None) -> bool:
    min_px = CAPTION_MIN_PX if min_px is None else min_px
    if min_px <= 0:
        return False
    size = image_size(img_path)
    return size is not None and min(size) < min_px


def _sniff_media_type(data: bytes) -> str | None:
    """Formato REAL de los bytes, por su firma.

    La extensión no es fiable (un .png que en realidad es JPEG pasa al extraer
    de algunos docx) y Anthropic rechaza la imagen si el media_type declarado no
    coincide con el contenido.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def prepare_image(img_path: Path, max_px=None) -> tuple[bytes, str]:
    """(bytes, media_type) listos para enviar al modelo.

    Reduce en memoria a `max_px` de lado mayor si hace falta, y transcodifica a
    PNG lo que no sea un formato aceptado. Nunca escribe en disco. Si pymupdf
    no puede decodificar la imagen (webp, formatos raros), se envía el original
    tal cual: mejor una llamada más cara que una imagen sin describir.
    """
    max_px = CAPTION_MAX_PX if max_px is None else max_px
    raw = Path(img_path).read_bytes()
    media = _sniff_media_type(raw)
    fallback = (raw, media or MEDIA_TYPES.get(Path(img_path).suffix.lower(), "image/png"))
    try:
        pymupdf = _pymupdf()
        pix = pymupdf.Pixmap(raw)
        grande = max_px > 0 and max(pix.width, pix.height) > max_px
        if media is not None and not grande:
            return raw, media  # sin cambios: se ahorra recomprimir
        # PNG/JPEG solo admiten gris o RGB (p. ej. JPEG CMYK de imprenta).
        if pix.colorspace is None or pix.colorspace.n not in (1, 3):
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        if grande:
            f = max_px / max(pix.width, pix.height)
            pix = pymupdf.Pixmap(pix, max(1, round(pix.width * f)),
                                 max(1, round(pix.height * f)), None)
        # Las fotos siguen en JPEG (en PNG pesarían varias veces más); el resto
        # —diagramas, capturas, cualquier cosa con transparencia— va en PNG.
        if media == "image/jpeg" and not pix.alpha:
            return pix.tobytes("jpg", jpg_quality=90), "image/jpeg"
        return pix.tobytes("png"), "image/png"
    except Exception:
        return fallback


def _content_hash(img_path: Path) -> str:
    return hashlib.sha256(Path(img_path).read_bytes()).hexdigest()


def _scan(output_dir: Path, min_px=None) -> tuple[list[Path], list[Path], int]:
    """(pendientes, diminutas_omitidas, total_encontradas)."""
    output_dir = Path(output_dir)
    todas = sorted(p for p in output_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    cache: dict[Path, dict[str, bool]] = {}
    pendientes, pequenas = [], []
    for img in todas:
        for md in md_files_for(img):
            estado = cache.get(md)
            if estado is None:
                estado = cache[md] = caption_state(md)
            if estado.get(img.name) is False:  # referenciada y sin caption
                # El tamaño se mira solo aquí, sobre las ya candidatas: decodificar
                # cada imagen del árbol en cada auditoría sería trabajo inútil.
                (pequenas if is_tiny(img, min_px) else pendientes).append(img)
                break
    return pendientes, pequenas, len(todas)


def pending_images(output_dir: Path, limit=None, min_px=None) -> tuple[list[Path], int]:
    """Imágenes que aún no tienen caption. Devuelve (pendientes, total_encontradas).

    Las diminutas (ver CAPTION_MIN_PX) no cuentan como pendientes.
    """
    pendientes, _, total = _scan(output_dir, min_px)
    if limit:
        pendientes = pendientes[:limit]
    return pendientes, total


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


# --------------------------------------------------------------------------- backend: Ollama

class OllamaError(RuntimeError):
    pass


def _ollama_post(path: str, payload: dict, host=None, timeout=None) -> dict:
    host = (host or OLLAMA_HOST).rstrip("/")
    req = urllib.request.Request(
        f"{host}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout or OLLAMA_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_models(host=None, timeout=15) -> list[str]:
    """Modelos disponibles en el servidor Ollama."""
    host = (host or OLLAMA_HOST).rstrip("/")
    with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m.get("name", "") for m in data.get("models", [])]


def check_ollama(model=DEFAULT_MODEL, host=None) -> None:
    """Falla temprano y con un mensaje accionable.

    Sin esto, un servidor inalcanzable o un modelo mal escrito se manifiestan
    como N errores idénticos, uno por imagen, a mitad de un lote largo.
    """
    host = (host or OLLAMA_HOST).rstrip("/")
    try:
        disponibles = ollama_models(host)
    except Exception as e:
        raise OllamaError(
            f"no se pudo contactar Ollama en {host}: {e}\n"
            f"       ¿Está corriendo? Pruébalo con: curl {host}/api/tags\n"
            f"       Desde un contenedor, OLLAMA_HOST debe apuntar al host "
            f"(http://host.docker.internal:11434), no a localhost."
        ) from e
    # Ollama admite el nombre sin tag (usa :latest); se acepta esa forma.
    if model not in disponibles and f"{model}:latest" not in disponibles \
            and not any(d.split(":")[0] == model for d in disponibles):
        raise OllamaError(
            f"el modelo '{model}' no está en {host}.\n"
            f"       Disponibles: {', '.join(disponibles) or '(ninguno)'}\n"
            f"       Descárgalo con: ollama pull {model}"
        )


def _caption_ollama(img_path: Path, model=DEFAULT_MODEL, host=None) -> str:
    raw, _ = prepare_image(img_path)
    data = base64.standard_b64encode(raw).decode("utf-8")
    body = _ollama_post("/api/chat", {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": PROMPT, "images": [data]}],
        # num_predict acota la respuesta: sin él, algunos modelos se explayan.
        "options": {"num_predict": 250, "temperature": 0.2},
    }, host=host)
    if body.get("error"):
        raise OllamaError(str(body["error"]))
    return (body.get("message") or {}).get("content", "").strip()


# --------------------------------------------------------------------------- backend: Anthropic

def _caption_anthropic(client, img_path: Path, model=DEFAULT_MODEL) -> str:
    # media_type sale de los bytes que se envían, no de la extensión del archivo.
    raw, media_type = prepare_image(img_path)
    data = base64.standard_b64encode(raw).decode("utf-8")
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


# --------------------------------------------------------------------------- despacho

def caption_image(img_path: Path, model=DEFAULT_MODEL, backend=None, client=None,
                  host=None, max_retries=DEFAULT_MAX_RETRIES) -> str:
    """Describe una imagen con el backend activo.

    El SDK de Anthropic ya reintenta solo; para Ollama (urllib pelado) el
    reintento con backoff se hace aquí: un servidor cargando el modelo en VRAM
    rechaza o tarda en las primeras peticiones del lote.
    """
    backend = (backend or CAPTION_BACKEND).lower()
    if backend == BACKEND_ANTHROPIC:
        return _caption_anthropic(client, img_path, model=model)

    ultimo = None
    for intento in range(max_retries):
        try:
            return _caption_ollama(img_path, model=model, host=host)
        except (urllib.error.URLError, OllamaError, TimeoutError, OSError) as e:
            ultimo = e
            if intento < max_retries - 1:
                time.sleep(min(2 ** intento, 30))
    raise OllamaError(f"tras {max_retries} intentos: {ultimo}")


def backend_disponible(backend=None) -> tuple[bool, str]:
    """(disponible, motivo). Sirve para que la web decida si ofrecer el paso."""
    backend = (backend or CAPTION_BACKEND).lower()
    if backend == BACKEND_ANTHROPIC:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False, "falta ANTHROPIC_API_KEY"
        return True, "API de Claude"
    try:
        check_ollama(DEFAULT_MODEL)
    except OllamaError as e:
        return False, str(e).splitlines()[0]
    return True, f"Ollama local ({DEFAULT_MODEL})"


def _group_by_content(images: list[Path]) -> list[list[Path]]:
    """Agrupa por sha256 del contenido, conservando el orden de aparición.

    Un logo repetido en cada página se extrae como N archivos idénticos con
    nombres distintos; describirlo N veces es pagar N llamadas por la misma
    respuesta. Si un archivo no se puede leer va solo en su grupo: el error
    saldrá al procesarlo, sin arrastrar a otras imágenes.
    """
    grupos: dict[str, list[Path]] = {}
    for img in images:
        try:
            clave = _content_hash(img)
        except OSError:
            clave = f"ilegible:{img}"
        grupos.setdefault(clave, []).append(img)
    return list(grupos.values())


def run(output_dir, model=DEFAULT_MODEL, workers=DEFAULT_WORKERS, limit=None,
        images=None, log=None, dry_run=False, backend=None, host=None,
        min_px=None) -> dict:
    """Genera e inserta captions. Devuelve un resumen con contadores."""
    backend = (backend or CAPTION_BACKEND).lower()
    min_px = CAPTION_MIN_PX if min_px is None else min_px

    def _log(msg):
        if log:
            log(msg)

    output_dir = Path(output_dir)
    if images is None:
        images, pequenas, total = _scan(output_dir, min_px)
        if limit:
            images = images[:limit]
    else:
        images = [Path(p) for p in images]
        total = len(images)
        tiny = {p: is_tiny(p, min_px) for p in images}
        pequenas = [p for p in images if tiny[p]]
        images = [p for p in images if not tiny[p]]

    grupos = _group_by_content(images)
    _log(f"Imágenes sin caption: {len(images)} (de {total} encontradas) "
         f"| llamadas al modelo: {len(grupos)} "
         f"| backend: {backend} | modelo: {model}")
    if len(grupos) < len(images):
        _log(f"Duplicadas por contenido: {len(images) - len(grupos)} imágenes "
             f"reutilizarán la descripción de una idéntica")
    if pequenas:
        _log(f"Omitidas por diminutas (lado menor < {min_px} px): {len(pequenas)}")
    resumen = {"pendientes": len(images), "total": total, "ok": 0, "errores": 0,
               "sin_referencia": 0, "duracion_s": 0.0, "modelo": model,
               "backend": backend, "llamadas": len(grupos), "reutilizadas": 0,
               "omitidas_pequenas": len(pequenas)}
    if dry_run or not images:
        return resumen

    client = None
    if backend == BACKEND_ANTHROPIC:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("falta ANTHROPIC_API_KEY (variable de entorno o .env)")
        import anthropic
        client = anthropic.Anthropic(max_retries=DEFAULT_MAX_RETRIES)
    else:
        # Preflight: mejor un error claro ahora que 400 fallos idénticos luego.
        check_ollama(model, host=host)
        _log(f"Ollama OK en {host or OLLAMA_HOST}")

    contador = threading.Lock()
    hechas = 0

    def process(grupo: list[Path]):
        nonlocal hechas
        rep = grupo[0]
        try:
            caption = caption_image(rep, model=model, backend=backend,
                                    client=client, host=host)
        except Exception as e:
            # Un fallo por imagen no aborta el lote: se registra y se sigue.
            # Reintentar el lote después es barato — lo ya hecho se salta solo.
            with contador:
                hechas += len(grupo)
                resumen["errores"] += len(grupo)
                extra = f" (y {len(grupo) - 1} idénticas)" if len(grupo) > 1 else ""
                _log(f"[ERROR] {rep.name}{extra}: {type(e).__name__}: {e}")
            return
        # La misma descripción para todas las copias. insert_caption toma el
        # lock de cada .md, así que esto es seguro aunque otro hilo esté
        # escribiendo en el mismo documento.
        for img in grupo:
            reutilizada = img is not rep
            try:
                inserted = any(insert_caption(md, img.name, caption) for md in md_files_for(img))
                fallo = None
            except Exception as e:
                inserted, fallo = False, e
            with contador:
                hechas += 1
                if fallo is not None:
                    resumen["errores"] += 1
                    _log(f"[ERROR] {img.name}: {type(fallo).__name__}: {fallo}")
                    continue
                if inserted:
                    resumen["ok"] += 1
                    if reutilizada:
                        resumen["reutilizadas"] += 1
                else:
                    resumen["sin_referencia"] += 1
                estado = "OK" if inserted else "sin referencia en .md"
                origen = f" (reutilizada de {rep.name})" if reutilizada else ""
                _log(f"[{hechas}/{len(images)}] {estado}: {img.name}{origen} -> {caption[:80]}")

    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(process, grupos))
    resumen["duracion_s"] = round(time.time() - started, 1)

    _log(f"Captions completados en {resumen['duracion_s']/60:.1f} min. "
         f"OK: {resumen['ok']} (reutilizadas: {resumen['reutilizadas']}) "
         f"| llamadas: {resumen['llamadas']} | sin referencia: {resumen['sin_referencia']} "
         f"| omitidas diminutas: {resumen['omitidas_pequenas']} | errores: {resumen['errores']}")
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
