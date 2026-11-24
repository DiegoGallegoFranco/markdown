"""API + interfaz web para subir documentos y recuperarlos como Markdown.

Arranque local:
    uvicorn webapp.main:app --host 0.0.0.0 --port 8000

Variables de entorno relevantes:
    DATA_DIR          dónde viven jobs, subidas y resultados (default: ./data)
    WORKERS           trabajos simultáneos (default: 1 — Marker satura la máquina)
    ANTHROPIC_API_KEY habilita el paso opcional de captioning de imágenes
    MAX_UPLOAD_MB     tope por archivo (default: 500)
    RETENTION_DAYS    borra jobs terminados más antiguos (default: 14; 0 = nunca)

SIN AUTENTICACIÓN: la app no pide credenciales. Cualquiera que alcance el
puerto puede subir documentos, leer los de otros y borrarlos. Publicarla solo
en localhost o en una red de confianza; si tiene que salir a internet, poner
delante un proxy inverso que resuelva el acceso (nginx/Caddy con auth y TLS,
o una VPN).
"""
from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from pipeline import captions as captions_mod
from pipeline.paths import SUPPORTED_EXTS, slugify
from webapp import jobs

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
CHUNK = 1024 * 1024

# --------------------------------------------------------------------------- estado del captioning
#
# backend_disponible() consulta Ollama por HTTP (timeout 15 s). Llamarlo en cada
# carga de la portada tiene dos costes: con Ollama caído cada visita tarda 15 s,
# y si se llama desde una ruta `async` bloquea el bucle de eventos entero —
# todas las demás peticiones (incluido el sondeo de progreso) esperan detrás.
# Se cachea unos segundos y siempre se llama desde un hilo.
CAPTIONS_STATUS_TTL_S = 30
_captions_status: tuple[float, tuple[bool, str]] | None = None
_captions_status_lock = threading.Lock()


def captions_status() -> tuple[bool, str]:
    """(disponible, motivo), cacheado CAPTIONS_STATUS_TTL_S. Bloqueante: no
    llamarlo directamente desde una ruta `async`."""
    global _captions_status
    # El lock serializa también la consulta: si Ollama está caído y llegan diez
    # visitas a la vez, solo una espera los 15 s; las demás reciben su resultado.
    with _captions_status_lock:
        ahora = time.monotonic()
        if _captions_status and ahora - _captions_status[0] < CAPTIONS_STATUS_TTL_S:
            return _captions_status[1]
        try:
            estado = captions_mod.backend_disponible()
        except Exception as e:  # nunca tumbar la portada por esto
            estado = (False, f"error comprobando el backend: {e}")
        _captions_status = (time.monotonic(), estado)
        return estado


@asynccontextmanager
async def lifespan(app: FastAPI):
    jobs.init_db()
    purgados = jobs.purge_old()
    jobs.start_purger()
    jobs.start_workers()
    reanudados = jobs.requeue_orphans()
    print(f"[webapp] listo (sin autenticación). workers={jobs.WORKERS} "
          f"reanudados={reanudados} purgados={purgados} "
          f"captions={captions_status()[1]}")
    yield
    jobs.stop_purger()


app = FastAPI(title="PDF/DOCX/PPTX → Markdown", lifespan=lifespan)


def _safe_name(filename: str) -> str:
    """Nunca confiar en el nombre que manda el navegador: podría traer `..` o
    rutas absolutas y escribir fuera del directorio del job."""
    name = Path(filename or "").name.strip()
    name = name.replace("\x00", "")
    if not name or name in (".", ".."):
        raise HTTPException(400, "Nombre de archivo inválido")
    return name


def _nombre_unico(name: str, usados: set[str]) -> str:
    """Evita que dos archivos del mismo lote se pisen.

    No basta con comparar el nombre tal cual: la carpeta de salida se deriva
    del nombre SANEADO (slugify), así que `informe final.pdf` e
    `informe_final.pdf` acabarían en la misma carpeta. Y en discos que no
    distinguen mayúsculas (macOS) `A.pdf` y `a.pdf` son el mismo archivo. Se
    compara por slug en minúsculas y se numera el repetido: `informe-2.pdf`.
    """
    p = Path(name)
    stem, n = p.stem, 1
    while (clave := slugify(stem).lower()) in usados:
        n += 1
        stem = f"{p.stem}-{n}"
    usados.add(clave)
    return f"{stem}{p.suffix}"


# --------------------------------------------------------------------------- vistas
#
# Las rutas son `def` (no `async def`) salvo la de subida: hacen I/O bloqueante
# (SQLite, leer logs, consultar Ollama) y FastAPI ejecuta las `def` en un pool
# de hilos. Una `async def` con I/O bloqueante congela el servidor entero.

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    captions_disp, captions_motivo = captions_status()
    # Starlette >= 0.29: el Request va primero; la forma antigua
    # TemplateResponse(nombre, {"request": ...}) ya no funciona.
    return TEMPLATES.TemplateResponse(request, "index.html", {
        "jobs": jobs.list_jobs(),
        # El captioning ya no depende de una clave: con backend ollama basta
        # con que el servidor responda y tenga el modelo.
        "captions_disponible": captions_disp,
        "captions_motivo": captions_motivo,
        "max_mb": jobs.MAX_UPLOAD_MB,
        "extensiones": sorted(SUPPORTED_EXTS),
        "cola": jobs.queue_depth(),
    })


@app.post("/jobs")
async def crear_job(
    request: Request,
    files: list[UploadFile],
    engine: str = Form("auto"),
    captions: str = Form(""),
):
    if engine not in ("auto", "pymupdf", "marker"):
        raise HTTPException(400, "Motor inválido")
    files = [f for f in files if f.filename]
    if not files:
        raise HTTPException(400, "No se subió ningún archivo")

    quiere_captions = captions in ("on", "true", "1", "yes")
    # Esta ruta sí es async (lee la subida en streaming): lo bloqueante, a un hilo.
    captions_ok = quiere_captions and (await run_in_threadpool(captions_status))[0]
    job_id = await run_in_threadpool(jobs.create_job, engine, captions_ok)
    input_dir = jobs.job_dir(job_id) / "input"

    guardados = 0
    usados: set[str] = set()
    for upload in files:
        name = _safe_name(upload.filename)
        if Path(name).suffix.lower() not in SUPPORTED_EXTS:
            continue
        name = _nombre_unico(name, usados)
        destino = input_dir / name
        escritos = 0
        with open(destino, "wb") as out:
            while chunk := await upload.read(CHUNK):
                escritos += len(chunk)
                if escritos > jobs.MAX_UPLOAD_MB * 1024 * 1024:
                    out.close()
                    destino.unlink(missing_ok=True)
                    jobs.delete_job(job_id)
                    raise HTTPException(413, f"{name} supera el límite de {jobs.MAX_UPLOAD_MB} MB")
                out.write(chunk)
        guardados += 1

    if guardados == 0:
        jobs.delete_job(job_id)
        raise HTTPException(400, "Ningún archivo con extensión soportada (.pdf, .docx, .pptx)")

    jobs.update(job_id, n_files=guardados)
    jobs.enqueue(job_id)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def ver_job(request: Request, job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")
    return TEMPLATES.TemplateResponse(request, "job.html", {
        "job": job, "log": jobs.read_log(job_id, tail=400),
    })


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")
    job["log"] = jobs.read_log(job_id, tail=400)
    return job


@app.get("/api/jobs")
def api_jobs():
    return {"jobs": jobs.list_jobs(), "cola": jobs.queue_depth()}


@app.get("/jobs/{job_id}/log", response_class=PlainTextResponse)
def ver_log(job_id: str):
    if jobs.get(job_id) is None:
        raise HTTPException(404, "Job no encontrado")
    return jobs.read_log(job_id)


@app.get("/jobs/{job_id}/download")
def descargar(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")
    zip_path = jobs.job_dir(job_id) / "resultado.zip"
    if not zip_path.exists():
        raise HTTPException(409, "El resultado todavía no está listo")
    return FileResponse(zip_path, media_type="application/zip",
                        filename=f"markdown-{job_id}.zip")


@app.post("/jobs/{job_id}/delete")
def borrar(job_id: str):
    # Borrar un job en ejecución dejaría al worker escribiendo en una carpeta
    # inexistente (y al subproceso de Marker trabajando para nadie). Los jobs
    # en cola sí se pueden borrar: el worker los descarta al no encontrarlos.
    if jobs.is_running(job_id):
        raise HTTPException(409, "El trabajo está en ejecución; espera a que termine para borrarlo")
    jobs.delete_job(job_id)
    return RedirectResponse("/", status_code=303)


@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    # Si el hilo del worker hubiera muerto, la cola dejaría de avanzar sin
    # error visible; se expone aquí para que un healthcheck lo detecte.
    vivos = jobs.workers_alive()
    if vivos < jobs.WORKERS:
        return PlainTextResponse(f"degradado: {vivos}/{jobs.WORKERS} workers vivos", status_code=503)
    return "ok"
