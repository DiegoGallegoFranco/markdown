"""Conversores documento -> Markdown, uno por formato/motor.

Todos comparten la misma firma y devuelven el mismo dict de resultado, para que
la CLI, el lote y el worker de la web los traten igual:

    convert_*(src, out_root, input_root, skip_existing=False, log=None) -> dict

Salida: `<out_root>/<estructura relativa>/<documento>/<documento>.md` con las
imágenes dentro de esa misma carpeta (`assets/`, o `media/` para pandoc, o
planas para Marker) y enlaces relativos al propio `.md`, de modo que la carpeta
de cada documento es autocontenida y portable.
"""
from __future__ import annotations

import codecs
import contextlib
import json
import os
import re
import selectors
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from .paths import doc_out_dir, slugify

# Marker es lento y ocasionalmente se cuelga en un PDF patológico; sin timeout
# un lote nocturno se queda bloqueado en un solo documento. El default es
# generoso a propósito: a ~15-30 s/página, un manual escaneado de varios
# cientos de páginas tarda horas legítimamente y no debe morir por el timeout.
MARKER_TIMEOUT_S = int(os.environ.get("MARKER_TIMEOUT_S", "14400"))
# Vacío = no pasar --mode y dejar que marker elija por dispositivo: `fast`
# (detectores CPU ligeros) en CPU/MPS, `balanced` (layout VLM + OCR de página
# completa) en GPU. Forzar `balanced` en una máquina sin GPU no solo es lento:
# requiere el binario llama-server, y sin él la conversión falla directamente.
# Valores válidos si se fija a mano: "balanced" | "fast".
MARKER_MODE = os.environ.get("MARKER_MODE", "").strip()


def _log(log, msg: str) -> None:
    if log:
        log(msg)


def _result(src, out_dir, md_path, engine, started, **extra) -> dict:
    md_path = Path(md_path) if md_path else None
    text = md_path.read_text(encoding="utf-8", errors="replace") if md_path and md_path.exists() else ""
    return {
        "ok": True,
        "engine": engine,
        "src": str(src),
        "out_dir": str(out_dir),
        "md": str(md_path) if md_path else None,
        "chars": len(text),
        "encabezados": sum(1 for line in text.splitlines() if line.strip().startswith("#")),
        "duracion_s": round(time.time() - started, 1),
        "error": None,
        **extra,
    }


def _error(src, engine, started, error) -> dict:
    return {
        "ok": False, "engine": engine, "src": str(src), "out_dir": None, "md": None,
        "chars": 0, "encabezados": 0, "imagenes": 0,
        "duracion_s": round(time.time() - started, 1), "error": str(error),
    }


def _existing_md(out_dir: Path) -> Path | None:
    """Devuelve el .md ya convertido en out_dir, si lo hay y no está vacío."""
    if not out_dir.is_dir():
        return None
    for md in sorted(out_dir.glob("*.md")):
        if md.stat().st_size > 0:
            return md
    return None


# --------------------------------------------------------------------------- PDF (pymupdf4llm)

def convert_pymupdf(src, out_root, input_root, skip_existing=False, log=None, dpi=150) -> dict:
    """PDF nativo -> Markdown. Rápido, solo CPU, sin modelos."""
    import pymupdf4llm

    started = time.time()
    src = Path(src)
    out_dir = doc_out_dir(src, out_root, input_root)
    try:
        if skip_existing and (md := _existing_md(out_dir)):
            _log(log, f"  SKIP (ya convertido): {src.name}")
            return _result(src, out_dir, md, "pymupdf", started,
                           imagenes=len(list((out_dir / "assets").glob("*"))), skipped=True)

        assets_dir = out_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        md_text = pymupdf4llm.to_markdown(
            str(src),
            write_images=True,
            image_path=str(assets_dir),
            image_format="png",
            dpi=dpi,
        )

        # pymupdf4llm incrusta los enlaces de imagen relativos al CWD, no al .md.
        # Se reescribe cualquier prefijo que termine en assets/ para que el
        # markdown sea portable si se mueve la carpeta o se abre suelto.
        md_text = re.sub(r'(!\[[^\]]*\]\()[^)]*?assets/([^)]+\))', r'\1assets/\2', md_text)

        md_path = out_dir / f"{slugify(src.stem)}.md"
        md_path.write_text(md_text, encoding="utf-8")

        n_images = len(list(assets_dir.glob("*")))
        _log(log, f"  OK ({round(time.time()-started,1)}s): {src.name} | {len(md_text)} chars | {n_images} img")
        return _result(src, out_dir, md_path, "pymupdf", started, imagenes=n_images, skipped=False)
    except Exception as e:
        _log(log, f"  ERROR pymupdf {src.name}: {e}")
        return _error(src, "pymupdf", started, e)


# --------------------------------------------------------------------------- PDF (Marker)

# ---- Sesión de Marker: el VLM se carga una vez por lote, no por documento ----
#
# En marker-pdf 2.0.0 / surya-ocr 0.22.1 los modelos no viven dentro de
# marker_single: viven en servidores aparte que surya lanza o reutiliza.
#
#   - Detectores del modo `fast` (rf-detr, ocr-error): surya ya los deja vivos
#     entre procesos. No se recargan por documento.
#   - VLM del modo `balanced` (llama-server): por defecto surya lo MATA al salir
#     cada marker_single (SURYA_INFERENCE_KEEP_ALIVE=False). En un lote, cada
#     documento volvía a arrancar llama-server y a subir el modelo a la GPU.
#
# Con keep-alive el siguiente marker_single se engancha al servidor que ya está
# corriendo (lo encuentra por un fichero "sentinel"). Pero dejarlo vivo para
# siempre retendría VRAM que el captioning con Ollama necesita justo después,
# así que se activa solo dentro de marker_session() y al cerrar la última
# sesión abierta se apaga el servidor — si lo arrancamos nosotros.

# Ruta interna de surya 0.22.1 (inference/backends/spawn.py:_sentinel_path).
# Se replica en vez de importarla porque importar surya arrastra torch en el
# proceso padre (la web), que nunca lo necesita. Versión pineada en
# requirements.txt; al subir surya, verificar que sigue siendo esta.
_SURYA_LLAMA_SENTINEL = Path("~/.cache/datalab/surya/llamacpp_server.json").expanduser()

_session_lock = threading.Lock()
_session_depth = 0          # sesiones abiertas (varios workers web a la vez)
_session_preexisting = False  # el servidor ya existía antes: no es nuestro, no se apaga


def _llama_server_pid() -> int | None:
    """PID del llama-server de surya si está vivo, según su sentinel."""
    try:
        pid = int(json.loads(_SURYA_LLAMA_SENTINEL.read_text())["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not _is_llama_server(pid):
        return None
    return pid


def _is_llama_server(pid: int) -> bool:
    """El PID sigue vivo Y es llama-server. Lo segundo protege contra un PID
    reciclado por el sistema: no matar un proceso ajeno por un sentinel viejo."""
    proc_cmd = Path(f"/proc/{pid}/cmdline")
    try:
        if proc_cmd.exists():  # Linux (el contenedor slim no trae `ps`)
            return b"llama-server" in proc_cmd.read_bytes()
        out = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                             capture_output=True, text=True, timeout=5)
        return "llama-server" in out.stdout
    except (OSError, subprocess.SubprocessError):
        return False


def _stop_llama_server(log=None) -> None:
    pid = _llama_server_pid()
    if pid is None:
        return
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(50):  # hasta 10 s para que libere la VRAM limpiamente
            time.sleep(0.2)
            if not _is_llama_server(pid):
                break
        else:
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as e:
        _log(log, f"  AVISO: no se pudo detener llama-server (pid {pid}): {e}")
        return
    with contextlib.suppress(OSError):
        _SURYA_LLAMA_SENTINEL.unlink()
    _log(log, "Marker: servidor VLM (llama-server) detenido; VRAM liberada.")


@contextlib.contextmanager
def marker_session(log=None):
    """Mantiene vivo el VLM de Marker entre documentos mientras dure el bloque.

    Reentrante y segura entre hilos: con WORKERS>1 dos jobs comparten el mismo
    servidor, y solo se apaga cuando se cierra la última sesión. Fuera de una
    sesión, marker_single se comporta como siempre (arranca y apaga el suyo).
    """
    global _session_depth, _session_preexisting
    with _session_lock:
        if _session_depth == 0:
            _session_preexisting = _llama_server_pid() is not None
        _session_depth += 1
    try:
        yield
    finally:
        with _session_lock:
            _session_depth -= 1
            if _session_depth == 0 and not _session_preexisting:
                _stop_llama_server(log)


def _marker_env() -> dict:
    env = os.environ.copy()
    # Con GPU NVIDIA surya elige por defecto el backend `vllm`, que intenta
    # lanzar un contenedor con `docker run`: dentro de nuestro contenedor no hay
    # docker, así que el modo `balanced` fallaría y el llama-server compilado
    # con CUDA (Dockerfile, WITH_LLAMA_SERVER) no se usaría nunca. Se fija
    # llama.cpp salvo que el entorno diga explícitamente otra cosa.
    env.setdefault("SURYA_INFERENCE_BACKEND", "llamacpp")
    with _session_lock:
        en_sesion = _session_depth > 0
    if en_sesion:
        env["SURYA_INFERENCE_KEEP_ALIVE"] = "true"
    return env


def _flatten_marker_output(out_dir: Path) -> None:
    """Aplana la subcarpeta redundante que crea marker_single.

    marker_single escribe en `<output_dir>/<nombre original>/<archivo>.md` — un
    nivel de más, y con el nombre SIN sanear aunque `--output_dir` sí lo esté.
    """
    for sub in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        if sub.name == "assets" or not any(sub.glob("*.md")):
            continue
        for item in sub.iterdir():
            destino = out_dir / item.name
            if destino.exists():
                shutil.rmtree(destino) if destino.is_dir() else destino.unlink()
            shutil.move(str(item), str(destino))
        sub.rmdir()


# Solo estos cuentan como "imágenes" de Marker: en la carpeta del documento
# también deja `<nombre>_meta.json`, que inflaba el conteo en uno por documento.
_MARKER_IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}

# Las barras de tqdm de marker se redibujan varias veces por segundo; volcarlas
# todas al log de la web lo llenaría de miles de líneas. Basta una cada tanto
# para que el usuario vea que avanza.
_MARKER_PROGRESS_EVERY_S = 15
_TQDM_RE = re.compile(r"\d+%\|")


def _count_marker_images(out_dir: Path) -> int:
    return sum(1 for p in out_dir.iterdir()
               if p.is_file() and p.suffix.lower() in _MARKER_IMG_EXTS)


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGTERM a todo el grupo de marker_single, 10 s de gracia, luego SIGKILL.

    Matar solo marker_single dejaba huérfanos a sus hijos (workers de torch,
    etc.) consumiendo CPU/GPU. El llama-server de surya NO cae aquí porque surya
    lo lanza con su propia sesión: lo apaga marker_session() al cerrar el lote.
    """
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # SIGKILL al grupo aunque el líder ya haya salido: un hijo que ignore
    # SIGTERM seguiría vivo con el mismo pgid.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


def _run_marker(cmd, log) -> tuple[int | None, list[str]]:
    """Ejecuta marker_single volcando su salida al log en streaming.

    Devuelve (returncode, últimas líneas); returncode None = timeout. Se lee con
    select en el propio hilo en vez de un hilo lector bloqueado en read(): si
    algún nieto heredara el pipe y siguiera vivo (p. ej. un servidor con
    keep-alive), un read() bloqueante no vería nunca EOF y colgaría el job.
    """
    # stderr unido a stdout: tqdm y los logs de marker van a stderr, y con dos
    # pipes habría que drenar ambos para no bloquear al hijo. start_new_session
    # crea un grupo propio para poder matar a marker y a todos sus hijos juntos.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, env=_marker_env(),
                            start_new_session=True)
    fd = proc.stdout.fileno()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    cola: deque[str] = deque(maxlen=5)
    buf = ""
    ultimo_progreso = 0.0
    progreso_pendiente: str | None = None

    def emitir(linea: str) -> None:
        nonlocal ultimo_progreso, progreso_pendiente
        linea = linea.strip()
        if not linea:
            return
        if _TQDM_RE.search(linea):
            ahora = time.monotonic()
            if ahora - ultimo_progreso >= _MARKER_PROGRESS_EVERY_S:
                ultimo_progreso = ahora
                progreso_pendiente = None
                _log(log, f"    marker: {linea}")
            else:
                progreso_pendiente = linea
            return
        # Las barras no van a la cola: el mensaje de error debe mostrar el
        # traceback/mensaje real, no el último redibujado de una barra.
        cola.append(linea)
        _log(log, f"    marker: {linea}")

    def consumir(chunk: bytes) -> None:
        nonlocal buf
        buf += decoder.decode(chunk)
        # tqdm redibuja con \r sin \n: partir solo por \n acumularía la barra
        # entera en una línea gigante que no se vería hasta el final.
        *lineas, buf = re.split(r"[\r\n]", buf)
        for linea in lineas:
            emitir(linea)

    sel = selectors.DefaultSelector()
    sel.register(fd, selectors.EVENT_READ)
    deadline = time.monotonic() + MARKER_TIMEOUT_S
    eof = False
    rc: int | None = None
    try:
        while True:
            if not eof:
                for _ in sel.select(timeout=1.0):
                    chunk = os.read(fd, 65536)
                    if chunk:
                        consumir(chunk)
                    else:
                        eof = True
                        sel.unregister(fd)
            else:
                # Sin nada que leer: esperar al proceso sin quemar CPU.
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=min(1.0, max(0.0, deadline - time.monotonic())))
            if proc.poll() is not None:
                rc = proc.returncode
                break
            if time.monotonic() >= deadline:
                _kill_process_group(proc)
                break
        # Drena lo que quede en el pipe tras la salida, con tope de tiempo por
        # si un nieto lo mantiene abierto y sigue escribiendo.
        fin_drenaje = time.monotonic() + 5
        while not eof and time.monotonic() < fin_drenaje:
            if not sel.select(timeout=0.2):
                break
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            consumir(chunk)
        consumir(b"")
        if buf:
            emitir(buf)
        if progreso_pendiente:  # el último estado de la barra (p. ej. 100%)
            _log(log, f"    marker: {progreso_pendiente}")
    finally:
        # Cualquier salida anómala (Ctrl-C en la CLI, excepción en `log`) no
        # debe dejar a marker corriendo: con start_new_session ya no recibe el
        # SIGINT de la terminal.
        if proc.poll() is None:
            _kill_process_group(proc)
        sel.close()
        proc.stdout.close()
    return rc, list(cola)


def convert_marker(src, out_root, input_root, skip_existing=False, log=None) -> dict:
    """PDF complejo o escaneado -> Markdown vía Marker (layout VLM + OCR completo).

    Mucho más lento que pymupdf4llm pero resuelve multicolumna, jerarquía de
    encabezados en documentos densos, y es el único de los dos que hace OCR real.
    """
    started = time.time()
    src = Path(src)
    out_dir = doc_out_dir(src, out_root, input_root)

    if skip_existing and (md := _existing_md(out_dir)):
        _log(log, f"  SKIP (ya convertido): {src.name}")
        return _result(src, out_dir, md, "marker", started,
                       imagenes=_count_marker_images(out_dir), skipped=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["marker_single", str(src), "--output_dir", str(out_dir)]
    if MARKER_MODE:
        cmd += ["--mode", MARKER_MODE]
    _log(log, f"  marker_single (modo: {MARKER_MODE or 'auto por dispositivo'}): "
              f"{src.name} (puede tardar minutos)")
    try:
        rc, cola = _run_marker(cmd, log)
    except FileNotFoundError:
        return _error(src, "marker", started,
                      "marker_single no está en el PATH (pip install marker-pdf)")
    if rc is None:
        _log(log, f"  ERROR marker {src.name}: timeout tras {MARKER_TIMEOUT_S}s")
        return _error(src, "marker", started, f"timeout tras {MARKER_TIMEOUT_S}s")

    if rc != 0:
        _log(log, f"  ERROR marker {src.name}: rc={rc} {' | '.join(cola)}")
        return _error(src, "marker", started, f"marker_single rc={rc}: {' '.join(cola)}")

    try:
        _flatten_marker_output(out_dir)
    except Exception as e:
        _log(log, f"  AVISO: no se pudo aplanar la salida de marker en {out_dir}: {e}")

    md = _existing_md(out_dir)
    if md is None:
        return _error(src, "marker", started, "marker terminó sin producir un .md no vacío")

    n_images = _count_marker_images(out_dir)
    _log(log, f"  OK ({round(time.time()-started,1)}s): {src.name} | {n_images} img")
    return _result(src, out_dir, md, "marker", started, imagenes=n_images, skipped=False)


# --------------------------------------------------------------------------- DOCX (pandoc)

def convert_docx(src, out_root, input_root, skip_existing=False, log=None) -> dict:
    """DOCX -> Markdown (GFM) con pandoc, extrayendo las imágenes embebidas.

    pandoc solo extrae imágenes referenciadas en el cuerpo (no encabezados ni
    pies): si `--extract-media` deja imágenes que no aparecen enlazadas en el
    .md, son decorativas.
    """
    started = time.time()
    src = Path(src)
    out_dir = doc_out_dir(src, out_root, input_root)

    if skip_existing and (md := _existing_md(out_dir)):
        _log(log, f"  SKIP (ya convertido): {src.name}")
        return _result(src, out_dir, md, "pandoc", started,
                       imagenes=len(list((out_dir / "media").rglob("*"))), skipped=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    md_name = f"{slugify(src.stem)}.md"
    md_path = out_dir / md_name
    # Se ejecuta pandoc DENTRO de out_dir con --extract-media=. : si se le pasa
    # una ruta absoluta, pandoc escribe esa ruta absoluta dentro del .md
    # (`<img src="/data/.../media/x.png">`) y la carpeta deja de ser portable —
    # muévela y las imágenes se rompen.
    cmd = ["pandoc", str(src.resolve()), "-f", "docx", "-t", "gfm",
           "--extract-media=.", "-o", md_name]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                              cwd=str(out_dir))
    except FileNotFoundError:
        return _error(src, "pandoc", started, "pandoc no está instalado (brew install pandoc)")
    except subprocess.TimeoutExpired:
        return _error(src, "pandoc", started, "timeout de pandoc (600s)")

    if proc.returncode != 0 or not md_path.exists():
        cola = (proc.stderr or "").strip().splitlines()[-3:]
        _log(log, f"  ERROR pandoc {src.name}: {' | '.join(cola)}")
        return _error(src, "pandoc", started, f"pandoc rc={proc.returncode}: {' '.join(cola)}")

    # Cinturón y tirantes: normaliza cualquier ruta que haya quedado absoluta o
    # con el prefijo "./", en las dos sintaxis que emite pandoc.
    text = md_path.read_text(encoding="utf-8", errors="replace")
    abs_prefix = re.escape(str(out_dir.resolve())) + r'/+'
    text = re.sub(abs_prefix, "", text)
    text = re.sub(r'(<img\b[^>]*?\bsrc\s*=\s*["\'])\./', r'\1', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'(!\[[^\]]*\]\()\./', r'\1', text)
    md_path.write_text(text, encoding="utf-8")

    n_images = len([p for p in (out_dir / "media").rglob("*") if p.is_file()]) if (out_dir / "media").exists() else 0
    _log(log, f"  OK ({round(time.time()-started,1)}s): {src.name} | {n_images} img")
    return _result(src, out_dir, md_path, "pandoc", started, imagenes=n_images, skipped=False)


# --------------------------------------------------------------------------- PPTX

# Formatos que un visor de Markdown puede renderizar. Los demás (emf/wmf, típicos
# de diagramas pegados desde Office) se extraen igual pero se enlazan como
# adjunto, no como imagen: un ![](x.emf) sale roto en cualquier visor y además
# ensucia el control de calidad con falsos enlaces rotos.
PPTX_RENDERABLE = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}


def _escape_cell(text: str) -> str:
    """Una celda con | o saltos de línea rompe la tabla Markdown entera."""
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", "<br>").strip()


def _iter_shapes(shapes):
    """Recorre las formas recursivamente: los grupos anidan contenido real.

    Sin esto se pierde en silencio todo el texto e imágenes dentro de un grupo,
    que en presentaciones reales es la mayoría del contenido.
    """
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from _iter_shapes(shape.shapes)
        else:
            yield shape


def convert_pptx(src, out_root, input_root, skip_existing=False, log=None) -> dict:
    """PPTX -> Markdown: texto por diapositiva, tablas, notas e imágenes extraídas."""
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    started = time.time()
    src = Path(src)
    out_dir = doc_out_dir(src, out_root, input_root)

    if skip_existing and (md := _existing_md(out_dir)):
        _log(log, f"  SKIP (ya convertido): {src.name}")
        return _result(src, out_dir, md, "pptx", started,
                       imagenes=len(list((out_dir / "assets").glob("*"))), skipped=True)

    try:
        assets_dir = out_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        prs = Presentation(str(src))
        md_lines = [f"# {src.stem}\n"]
        img_count = 0

        for i, slide in enumerate(prs.slides, start=1):
            md_lines.append(f"## Diapositiva {i}\n")
            title_shape = slide.shapes.title

            for shape in _iter_shapes(slide.shapes):
                if title_shape is not None and shape == title_shape:
                    if shape.text.strip():
                        md_lines.append(f"**{shape.text.strip()}**\n")
                    continue

                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    img_count += 1
                    ext = (shape.image.ext or "bin").lower()
                    img_name = f"slide{i:03d}-{img_count:03d}.{ext}"
                    (assets_dir / img_name).write_bytes(shape.image.blob)
                    if ext in PPTX_RENDERABLE:
                        md_lines.append(f"![](assets/{img_name})\n")
                    else:
                        md_lines.append(f"[archivo adjunto: {img_name}](assets/{img_name})\n")

                elif shape.has_table:
                    rows = [[_escape_cell(c.text) for c in row.cells] for row in shape.table.rows]
                    if rows:
                        md_lines.append("| " + " | ".join(rows[0]) + " |")
                        md_lines.append("|" + "---|" * len(rows[0]))
                        for r in rows[1:]:
                            md_lines.append("| " + " | ".join(r) + " |")
                        md_lines.append("")

                elif shape.has_text_frame:
                    # para.text (no join de runs): un párrafo sin runs explícitos
                    # todavía puede tener texto.
                    lines = [p.text.strip() for p in shape.text_frame.paragraphs if p.text.strip()]
                    for line in lines:
                        md_lines.append(f"- {line}")
                    if lines:
                        md_lines.append("")

            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    md_lines.append(f"> Notas: {notes}\n")
            md_lines.append("")

        md_text = "\n".join(md_lines)
        md_path = out_dir / f"{slugify(src.stem)}.md"
        md_path.write_text(md_text, encoding="utf-8")

        _log(log, f"  OK ({round(time.time()-started,1)}s): {src.name} | {len(prs.slides)} diapositivas | {img_count} img")
        return _result(src, out_dir, md_path, "pptx", started, imagenes=img_count, skipped=False)
    except Exception as e:
        _log(log, f"  ERROR pptx {src.name}: {e}")
        return _error(src, "pptx", started, e)


# --------------------------------------------------------------------------- despacho

CONVERTERS = {
    "pymupdf": convert_pymupdf,
    "marker": convert_marker,
    "pandoc": convert_docx,
    "pptx": convert_pptx,
}


def engine_for(src, pdf_engine="pymupdf") -> str:
    ext = Path(src).suffix.lower()
    if ext == ".pdf":
        return pdf_engine
    if ext == ".docx":
        return "pandoc"
    if ext == ".pptx":
        return "pptx"
    raise ValueError(f"Formato no soportado: {ext}")


def convert(src, out_root, input_root, pdf_engine="pymupdf", skip_existing=False, log=None) -> dict:
    engine = engine_for(src, pdf_engine)
    return CONVERTERS[engine](src, out_root, input_root, skip_existing=skip_existing, log=log)


def convert_batch(sources, out_root, input_root=None, pdf_engine="pymupdf",
                  skip_existing=False, log=None, on_result=None) -> list[dict]:
    """Convierte un lote. Un fallo NO detiene el resto: se registra y se sigue."""
    from .paths import common_root

    sources = [Path(s) for s in sources]
    input_root = Path(input_root) if input_root else common_root(sources)
    results = []
    with marker_session(log):
        for n, src in enumerate(sources, start=1):
            _log(log, f"[{n}/{len(sources)}] {src.name}")
            try:
                r = convert(src, out_root, input_root, pdf_engine=pdf_engine,
                            skip_existing=skip_existing, log=log)
            except ValueError as e:  # formato no soportado
                r = _error(src, "?", time.time(), e)
                _log(log, f"  ERROR: {e}")
            results.append(r)
            if on_result:
                on_result(r)
    return results
