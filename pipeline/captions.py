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

Dos modos (CAPTION_MODE):

    detallado  (default) el modelo identifica el tipo de imagen y EXTRAE su
               contenido (tablas completas, transcripciones, relaciones de un
               diagrama...) en un bloque estructurado bajo la imagen, y se
               genera un índice de imágenes. Ver pipeline/extraccion.py.
    breve      la descripción de 1-2 frases de siempre.

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

from . import extraccion as X
from . import indice
from .mdimages import (MARKER_BREVE, es_descripcion_breve, es_descripcion_ia,
                       image_refs, ref_name, reference_pattern)

# Formatos que aceptan ambos backends.
IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
MEDIA_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
}

MARKER = MARKER_BREVE  # compatibilidad: formato breve

MODO_DETALLADO, MODO_BREVE = "detallado", "breve"
CAPTION_MODE = (os.environ.get("CAPTION_MODE") or MODO_DETALLADO).strip().lower()
if CAPTION_MODE not in (MODO_DETALLADO, MODO_BREVE):
    CAPTION_MODE = MODO_DETALLADO

def _env_int(nombre: str, defecto: str) -> int:
    """Entero de entorno; vacío cuenta como no definido. docker-compose pasa
    `${VAR:-}` como cadena vacía cuando no está en .env, e int("") reventaría
    al importar el módulo (y con él la web entera)."""
    valor = (os.environ.get(nombre) or "").strip()
    return int(valor) if valor else int(defecto)


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
DEFAULT_WORKERS = _env_int("CAPTION_WORKERS", "1" if CAPTION_BACKEND == BACKEND_OLLAMA else "8")
DEFAULT_MAX_RETRIES = _env_int("CAPTION_MAX_RETRIES", "6")

# Umbral de "imagen diminuta", sobre el LADO MENOR en píxeles. Se usa el lado
# menor y no el área porque así caen tanto los iconos/viñetas (16x16, 32x32)
# como las líneas separadoras y filetes (800x3), que tienen área grande pero
# ningún contenido que describir. 0 desactiva el filtro.
CAPTION_MIN_PX = _env_int("CAPTION_MIN_PX", "64")
# Lado mayor máximo de lo que se ENVÍA al modelo. El coste y el tiempo de un VLM
# crecen con la resolución: para una descripción de 1-2 frases 1024 px sobra,
# pero para LEER las cifras de una tabla densa hace falta más (1600 en modo
# detallado). Se reduce solo en memoria: el archivo en disco es el que enlaza
# el .md y no se toca. 0 desactiva la reducción.
CAPTION_MAX_PX = _env_int("CAPTION_MAX_PX", "1600" if CAPTION_MODE == MODO_DETALLADO else "1024")
# Modo detallado: tope de la respuesta (una tabla completa necesita bastante
# más que una frase) y ventana de contexto. Ollama usa por defecto una ventana
# pequeña; una imagen de 1600 px son miles de tokens en qwen-vl, y con la
# ventana por defecto la imagen o la respuesta se truncarían sin error visible.
CAPTION_NUM_PREDICT = _env_int("CAPTION_NUM_PREDICT", "2048")
CAPTION_NUM_CTX = _env_int("CAPTION_NUM_CTX", "8192")
# Caché persistente por contenido (modo detallado): una imagen idéntica ya
# extraída en otro trabajo no vuelve a pasar por el modelo. "" la desactiva.
CAPTION_CACHE_DIR = os.environ.get("CAPTION_CACHE_DIR")
CAPTION_CACHE_DIR = (Path(CAPTION_CACHE_DIR) if CAPTION_CACHE_DIR
                     else (None if CAPTION_CACHE_DIR == "" else X.dir_cache_por_defecto()))

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
            estado[name] = i + 1 < len(paras) and es_descripcion_ia(paras[i + 1])
    return estado


def md_files_for(img_path: Path) -> list[Path]:
    """Los .md que pueden referenciar esta imagen (misma carpeta o su padre)."""
    doc_dir = img_path.parent
    if doc_dir.name in ("assets", "media"):
        doc_dir = doc_dir.parent
    return sorted(md for md in doc_dir.glob("*.md") if md.name != "INDICE_IMAGENES.md")


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


def _insertar(md_path: Path, img_name: str, bloque_de) -> bool:
    """Inserta bajo la primera aparición de la imagen el texto que devuelva
    `bloque_de(texto_actual_del_md)`. Idempotente y con el lock del .md."""
    pattern = reference_pattern(img_name)
    with _lock_for(md_path):
        # Se relee dentro del lock: otro hilo pudo haber escrito ya.
        if caption_state(md_path).get(img_name, True):
            return False
        text = md_path.read_text(encoding="utf-8")
        marca = "\n\n" + bloque_de(text)
        # m.group(0): la etiqueta completa, sea ![](x) o <img ...> — esta última
        # puede ocupar varias líneas, así que hay que conservarla entera.
        new_text, n = pattern.subn(lambda m: m.group(0) + marca, text, count=1)
        if n:
            md_path.write_text(new_text, encoding="utf-8")
        return n > 0


def insert_caption(md_path: Path, img_name: str, caption: str) -> bool:
    """Modo breve: inserta `*[descripción generada por IA: ...]*`. Idempotente."""
    return _insertar(md_path, img_name, lambda _t: f"*[descripción generada por IA: {caption}]*")


def insert_extraccion(md_path: Path, img_path: Path, datos: dict, sha256: str,
                      modelo: str, reutilizada_de: str | None = None) -> bool:
    """Modo detallado: inserta el bloque estructurado y registra la imagen en el
    imagenes.json del documento. Idempotente.

    El número, la sección y el id se calculan con el .md tal como está DENTRO
    del lock, así que son deterministas aunque varios hilos escriban a la vez.
    """
    md_path, img_path = Path(md_path), Path(img_path)
    registro = {}

    def bloque_de(texto: str) -> str:
        ctx = X.contexto(texto, img_path.name)
        pagina, unidad = X.pagina_de(img_path.name)
        ident = f"{md_path.stem}/img-{ctx['numero']:03d}"
        registro.update({
            "id": ident, "numero": ctx["numero"], "documento": md_path.stem,
            "md": md_path.name, "archivo": img_path.relative_to(md_path.parent).as_posix()
            if img_path.is_relative_to(md_path.parent) else img_path.name,
            "pagina": pagina, "unidad_pagina": unidad, "seccion": ctx["seccion"],
            **datos, "sha256": sha256, "modelo": modelo,
            "prompt_version": X.PROMPT_VERSION, "reutilizada_de": reutilizada_de,
        })
        return X.render_bloque(datos, ctx["numero"], ident, pagina, unidad, ctx["seccion"])

    insertado = _insertar(md_path, img_path.name, bloque_de)
    if insertado:
        with _lock_for(md_path):
            indice.upsert_sidecar(md_path.parent, registro)
    return insertado


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


def _extraer_ollama(img_path: Path, prompt: str, model=DEFAULT_MODEL, host=None) -> str:
    raw, _ = prepare_image(img_path)
    data = base64.standard_b64encode(raw).decode("utf-8")
    body = _ollama_post("/api/chat", {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt, "images": [data]}],
        # `format` con JSON Schema: Ollama restringe la salida a ese esquema.
        "format": X.ESQUEMA,
        # Temperatura casi nula: se pide transcribir, no redactar.
        "options": {"num_predict": CAPTION_NUM_PREDICT, "num_ctx": CAPTION_NUM_CTX,
                    "temperature": 0.1},
    }, host=host)
    if body.get("error"):
        raise OllamaError(str(body["error"]))
    return (body.get("message") or {}).get("content", "")


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


def _extraer_anthropic(client, img_path: Path, prompt: str, model=DEFAULT_MODEL) -> str:
    raw, media_type = prepare_image(img_path)
    data = base64.standard_b64encode(raw).decode("utf-8")
    claves = ", ".join(X.ESQUEMA["required"])
    resp = client.messages.create(
        model=model,
        max_tokens=CAPTION_NUM_PREDICT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                {"type": "text", "text": f"{prompt}\n\nClaves del JSON: {claves}. "
                                         f"Valores de tipo: {', '.join(X.TIPOS)}."},
            ],
        }],
    )
    if resp.stop_reason == "refusal":
        return json.dumps({"tipo": "otro", "titulo": "", "resumen":
                           "El modelo declinó procesar esta imagen.", "calidad": "baja"})
    return next((b.text for b in resp.content if b.type == "text"), "")


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


def extraer_imagen(img_path: Path, ctx: dict, model=DEFAULT_MODEL, backend=None,
                   client=None, host=None, max_retries=DEFAULT_MAX_RETRIES) -> tuple[dict, bool]:
    """Modo detallado: (datos normalizados, ok). Mismos reintentos que caption_image."""
    backend = (backend or CAPTION_BACKEND).lower()
    prompt = X.prompt_con_contexto(ctx)
    if backend == BACKEND_ANTHROPIC:
        return X.interpretar(_extraer_anthropic(client, img_path, prompt, model=model))
    ultimo = None
    for intento in range(max_retries):
        try:
            return X.interpretar(_extraer_ollama(img_path, prompt, model=model, host=host))
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
        min_px=None, modo=None) -> dict:
    """Genera e inserta captions. Devuelve un resumen con contadores."""
    backend = (backend or CAPTION_BACKEND).lower()
    modo = (modo or CAPTION_MODE).lower()
    detallado = modo == MODO_DETALLADO
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
         f"| backend: {backend} | modelo: {model} | modo: {modo}")
    if len(grupos) < len(images):
        _log(f"Duplicadas por contenido: {len(images) - len(grupos)} imágenes "
             f"reutilizarán la descripción de una idéntica")
    if pequenas:
        _log(f"Omitidas por diminutas (lado menor < {min_px} px): {len(pequenas)}")
    resumen = {"pendientes": len(images), "total": total, "ok": 0, "errores": 0,
               "sin_referencia": 0, "duracion_s": 0.0, "modelo": model,
               "backend": backend, "llamadas": len(grupos), "reutilizadas": 0,
               "omitidas_pequenas": len(pequenas), "modo": modo,
               "desde_cache": 0, "extraccion_incompleta": 0}
    if dry_run:
        return resumen
    if not images:
        if detallado:
            _indexar(output_dir, _log)
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
            if detallado:
                datos, sha, origen_cache = _extraer_grupo(rep, model, backend, client, host)
                caption = f"[{datos['tipo']}] {datos['titulo']}"
            else:
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
        if detallado:
            with contador:
                if origen_cache == "cache":
                    resumen["desde_cache"] += 1
                elif origen_cache == "incompleta":
                    resumen["extraccion_incompleta"] += 1
        for img in grupo:
            reutilizada = img is not rep
            try:
                if detallado:
                    inserted = any(insert_extraccion(
                        md, img, datos, sha, model,
                        reutilizada_de=rep.name if reutilizada else None)
                        for md in md_files_for(img))
                else:
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
    if detallado:
        if resumen["desde_cache"]:
            _log(f"Extracciones tomadas de la caché (sin llamar al modelo): {resumen['desde_cache']}")
        if resumen["extraccion_incompleta"]:
            _log(f"AVISO: {resumen['extraccion_incompleta']} extracción(es) con respuesta no "
                 f"válida; se guardó el texto recibido con calidad baja. Si son tablas muy "
                 f"grandes, subir CAPTION_NUM_PREDICT.")
        _indexar(output_dir, _log)
    return resumen


def _extraer_grupo(rep: Path, model, backend, client, host) -> tuple[dict, str, str]:
    """(datos, sha256, origen) para el representante de un grupo de idénticas.
    origen: "cache" | "modelo" | "incompleta"."""
    sha = _content_hash(rep)
    clave = X.clave_cache(sha, model, CAPTION_MAX_PX)
    if (en_cache := X.cache_get(CAPTION_CACHE_DIR, clave)) is not None:
        return X.normalizar(en_cache), sha, "cache"
    # El contexto sale del primer .md que referencia la imagen. Las copias
    # idénticas en otros documentos comparten la extracción (son la misma
    # imagen); su número, sección e id sí son los de cada documento.
    ctx = {}
    for md in md_files_for(rep):
        try:
            ctx = X.contexto(md.read_text(encoding="utf-8", errors="replace"), rep.name)
            break
        except OSError:
            continue
    datos, ok = extraer_imagen(rep, ctx, model=model, backend=backend, client=client, host=host)
    if ok:
        X.cache_put(CAPTION_CACHE_DIR, clave, datos)
    return datos, sha, ("modelo" if ok else "incompleta")


def _indexar(output_dir: Path, _log) -> None:
    try:
        r = indice.construir(output_dir)
        if r["imagenes"]:
            _log(f"Índice de imágenes: {r['imagenes']} imágenes de {r['documentos']} "
                 f"documento(s) -> {Path(r['rutas'][0]).name}, indice_imagenes.csv/.json")
    except Exception as e:  # el índice es un derivado: nunca tumba el paso
        _log(f"AVISO: no se pudo generar el índice de imágenes: {type(e).__name__}: {e}")


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
        paras = text.split("\n\n")
        if not any(es_descripcion_ia(p) for p in paras):
            continue
        out, removed, i = [], 0, 0
        while i < len(paras):
            out.append(paras[i])
            if es_descripcion_ia(paras[i]):
                j = i + 1
                while j < len(paras) and es_descripcion_ia(paras[j]):
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


def quitar_descripciones(output_dir, solo_breves=True, log=None) -> dict:
    """Quita los bloques de IA para volver a generarlos (p. ej. pasar a modo
    detallado documentos ya descritos en modo breve). Con solo_breves=False
    quita también los detallados y sus imagenes.json; la caché se conserva, así
    que rehacer una imagen ya extraída no vuelve a llamar al modelo."""
    quitados = archivos = 0
    for md in sorted(Path(output_dir).rglob("*.md")):
        if md.name == "INDICE_IMAGENES.md":
            continue
        text = md.read_text(encoding="utf-8", errors="replace")
        paras = text.split("\n\n")
        criterio = es_descripcion_breve if solo_breves else es_descripcion_ia
        nuevos = [p for p in paras if not criterio(p)]
        if len(nuevos) != len(paras):
            with _lock_for(md):
                md.write_text("\n\n".join(nuevos), encoding="utf-8")
                if not solo_breves:
                    (md.parent / indice.SIDECAR).unlink(missing_ok=True)
            quitados += len(paras) - len(nuevos)
            archivos += 1
            if log:
                log(f"{md}: -{len(paras) - len(nuevos)} descripciones")
    return {"quitadas": quitados, "archivos_modificados": archivos}
