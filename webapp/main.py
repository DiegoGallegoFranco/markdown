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

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from pipeline import captions as captions_mod
from pipeline import converters as converters_mod
from pipeline.paths import SUPPORTED_EXTS
from webapp import jobs

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
CHUNK = 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    jobs.init_db()
    purgados = jobs.purge_old()
    jobs.start_workers()
    reanudados = jobs.requeue_orphans()
    # flush explícito: fuera de Docker (donde el Dockerfile fija
    # PYTHONUNBUFFERED=1) stdout va bloque-bufferizado y el diagnóstico de
    # arranque no aparecería hasta llenar el búfer.
    print(f"[webapp] listo (sin autenticación). workers={jobs.WORKERS} "
          f"reanudados={reanudados} purgados={purgados}", flush=True)
    # Se dejan en el log al arrancar: una GPU que no se ve degrada la
    # conversión en silencio, y un Ollama inalcanzable solo se descubriría al
    # subir el primer documento.
    print(f"[webapp] conversión: {converters_mod.device_summary()}", flush=True)
    print(f"[webapp] captions:   {captions_mod.backend_disponible()[1]}", flush=True)
    yield


app = FastAPI(title="PDF/DOCX/PPTX → Markdown", lifespan=lifespan)


def _safe_name(filename: str) -> str:
    """Nunca confiar en el nombre que manda el navegador: podría traer `..` o
    rutas absolutas y escribir fuera del directorio del job."""
    name = Path(filename or "").name.strip()
    name = name.replace("\x00", "")
    if not name or name in (".", ".."):
        raise HTTPException(400, "Nombre de archivo inválido")
    return name


# --------------------------------------------------------------------------- vistas

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    captions_disp, captions_motivo = captions_mod.backend_disponible()
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
    job_id = jobs.create_job(engine, quiere_captions and captions_mod.backend_disponible()[0])
    input_dir = jobs.job_dir(job_id) / "input"

    guardados = 0
    for upload in files:
        name = _safe_name(upload.filename)
        if Path(name).suffix.lower() not in SUPPORTED_EXTS:
            continue
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
async def ver_job(request: Request, job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")
    return TEMPLATES.TemplateResponse(request, "job.html", {
        "job": job, "log": jobs.read_log(job_id, tail=400),
    })


@app.get("/api/jobs/{job_id}")
async def api_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")
    job["log"] = jobs.read_log(job_id, tail=400)
    return job


@app.get("/api/jobs")
async def api_jobs():
    return {"jobs": jobs.list_jobs(), "cola": jobs.queue_depth()}


@app.get("/jobs/{job_id}/log", response_class=PlainTextResponse)
async def ver_log(job_id: str):
    if jobs.get(job_id) is None:
        raise HTTPException(404, "Job no encontrado")
    return jobs.read_log(job_id)


@app.get("/jobs/{job_id}/download")
async def descargar(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")
    zip_path = jobs.job_dir(job_id) / "resultado.zip"
    if not zip_path.exists():
        raise HTTPException(409, "El resultado todavía no está listo")
    return FileResponse(zip_path, media_type="application/zip",
                        filename=f"markdown-{job_id}.zip")


@app.post("/jobs/{job_id}/delete")
async def borrar(job_id: str):
    jobs.delete_job(job_id)
    return RedirectResponse("/", status_code=303)


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz():
    return "ok"
