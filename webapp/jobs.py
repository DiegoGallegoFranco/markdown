"""Cola de trabajos y almacén de estado para la interfaz web.

Deliberadamente simple: SQLite en disco + un pool de hilos en el propio proceso.
No hay Redis ni Celery porque el cuello de botella real es Marker (un documento
a la vez satura la GPU/CPU), así que una cola distribuida no compraría nada y sí
añadiría dos servicios que mantener.

Cada job es un lote de documentos subidos juntos y vive en su propio directorio:

    <DATA_DIR>/jobs/<job_id>/
        input/      documentos subidos
        output/     markdown + imágenes (una carpeta por documento)
        job.log     traza completa
        resultado.zip
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import sqlite3
import threading
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pipeline import captions as captions_mod
from pipeline import converters, inventory, qc
from pipeline.paths import SUPPORTED_EXTS, collect_documents, slugify

DATA_DIR = Path(os.environ.get("DATA_DIR", "data")).resolve()
DB_PATH = DATA_DIR / "jobs.db"
JOBS_DIR = DATA_DIR / "jobs"
WORKERS = int(os.environ.get("WORKERS", "1"))
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "14"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "500"))

STATUS_QUEUED, STATUS_RUNNING, STATUS_DONE, STATUS_ERROR = "queued", "running", "done", "error"

_schema = """
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT,
    status       TEXT NOT NULL,
    engine       TEXT NOT NULL,
    captions     INTEGER NOT NULL DEFAULT 0,
    n_files      INTEGER NOT NULL DEFAULT 0,
    n_done       INTEGER NOT NULL DEFAULT 0,
    step         TEXT,
    error        TEXT,
    summary      TEXT
);
"""

_queue: "queue.Queue[str]" = queue.Queue()
_workers_started = threading.Event()


# --------------------------------------------------------------------------- almacén

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    # WAL: permite que el worker escriba progreso mientras la web lee el estado.
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(_schema)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def create_job(engine: str, captions: bool) -> str:
    job_id = uuid.uuid4().hex[:12]
    d = job_dir(job_id)
    (d / "input").mkdir(parents=True, exist_ok=True)
    (d / "output").mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, created_at, status, engine, captions) VALUES (?,?,?,?,?)",
            (job_id, _now(), STATUS_QUEUED, engine, int(captions)),
        )
    return job_id


def update(job_id: str, **fields) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    with _connect() as conn:
        conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))


def get(job_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["captions"] = bool(d["captions"])
    d["summary"] = json.loads(d["summary"]) if d["summary"] else None
    return d


def list_jobs(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d["captions"] = bool(d["captions"])
        d["summary"] = json.loads(d["summary"]) if d["summary"] else None
        out.append(d)
    return out


def delete_job(job_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    shutil.rmtree(job_dir(job_id), ignore_errors=True)
    return cur.rowcount > 0


def read_log(job_id: str, tail: int | None = None) -> str:
    log_path = job_dir(job_id) / "job.log"
    if not log_path.exists():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if tail:
        return "\n".join(text.splitlines()[-tail:])
    return text


def _append_log(job_id: str, msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    with open(job_dir(job_id) / "job.log", "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {msg}\n")


# --------------------------------------------------------------------------- ejecución

def _run_job(job_id: str) -> None:
    d = job_dir(job_id)
    input_dir, output_dir = d / "input", d / "output"
    job = get(job_id)
    if job is None:
        return

    def log(msg):
        _append_log(job_id, str(msg))

    update(job_id, status=STATUS_RUNNING, started_at=_now(), step="inventario", error=None)
    try:
        docs = collect_documents(input_dir, exts=SUPPORTED_EXTS)
        if not docs:
            raise RuntimeError("no se subió ningún documento con extensión soportada "
                               "(.pdf, .docx, .pptx)")
        update(job_id, n_files=len(docs))
        log(f"{len(docs)} documento(s): {', '.join(p.name for p in docs)}")

        # Elección de motor por documento. `auto` clasifica cada PDF primero:
        # los escaneados necesitan el OCR real de Marker, los nativos salen
        # bien y órdenes de magnitud más rápido con pymupdf4llm.
        pdfs = [p for p in docs if p.suffix.lower() == ".pdf"]
        engines: dict[Path, str] = {}
        if job["engine"] == "auto" and pdfs:
            rows = inventory.build_inventory(pdfs, input_dir, on_progress=log)
            inventory.write_csv(rows, output_dir / "inventario.csv")
            engines = {p: r.get("motor_sugerido") or "pymupdf" for p, r in zip(pdfs, rows)}
        else:
            engines = {p: (job["engine"] if job["engine"] != "auto" else "pymupdf") for p in pdfs}

        update(job_id, step="conversion")
        results = []
        for n, src in enumerate(docs, start=1):
            log(f"[{n}/{len(docs)}] {src.name} (motor: {engines.get(src, converters.engine_for(src))})")
            r = converters.convert(src, output_dir, input_dir,
                                   pdf_engine=engines.get(src, "pymupdf"),
                                   skip_existing=True, log=log)
            results.append(r)
            update(job_id, n_done=n)

        # Se indexa por el nombre SANEADO: el .md se llama como el slug, no
        # como el archivo original, así que cruzar por el nombre de entrada
        # dejaría la columna de duración vacía.
        durations = {slugify(Path(r["src"]).stem): r["duracion_s"] for r in results if r["ok"]}

        if job["captions"]:
            update(job_id, step="captions")
            if not os.environ.get("ANTHROPIC_API_KEY"):
                log("AVISO: se pidió captioning pero falta ANTHROPIC_API_KEY. Paso omitido.")
            else:
                log("Generando descripciones de imágenes...")
                captions_mod.run(output_dir, log=log)

        update(job_id, step="qc")
        rows = qc.run(output_dir, root=output_dir, durations=durations)
        qc.write_csv(rows, output_dir / "inventario_final.csv")
        resumen = qc.summarize(rows)
        resumen["convertidos"] = sum(1 for r in results if r["ok"])
        resumen["fallidos"] = sum(1 for r in results if not r["ok"])
        resumen["errores"] = [
            {"documento": Path(r["src"]).name, "error": r["error"]}
            for r in results if not r["ok"]
        ]
        log(f"QC: {resumen['ok']} OK, {resumen['revision']} para revisión, "
            f"{resumen['error']} con error")

        update(job_id, step="empaquetado")
        shutil.make_archive(str(d / "resultado"), "zip", root_dir=output_dir)
        log("resultado.zip generado.")

        update(job_id, status=STATUS_DONE, finished_at=_now(), step=None,
               summary=json.dumps(resumen, ensure_ascii=False))
    except Exception as e:
        _append_log(job_id, f"ERROR FATAL: {e}\n{traceback.format_exc()}")
        update(job_id, status=STATUS_ERROR, finished_at=_now(), step=None, error=str(e))


def _worker_loop() -> None:
    while True:
        job_id = _queue.get()
        try:
            _run_job(job_id)
        finally:
            _queue.task_done()


def enqueue(job_id: str) -> None:
    _queue.put(job_id)


def queue_depth() -> int:
    return _queue.qsize()


def start_workers() -> None:
    if _workers_started.is_set():
        return
    _workers_started.set()
    for i in range(WORKERS):
        threading.Thread(target=_worker_loop, name=f"pipeline-worker-{i}", daemon=True).start()


def requeue_orphans() -> int:
    """Reencola los jobs que quedaron 'running' al reiniciar el proceso.

    Es seguro: la conversión usa skip_existing, así que reanuda desde donde
    estaba en vez de rehacer los documentos ya convertidos.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id FROM jobs WHERE status IN (?, ?) ORDER BY created_at",
            (STATUS_RUNNING, STATUS_QUEUED),
        ).fetchall()
    for row in rows:
        _append_log(row["id"], "Reanudando tras reinicio del servicio.")
        update(row["id"], status=STATUS_QUEUED)
        enqueue(row["id"])
    return len(rows)


def purge_old(days: int = RETENTION_DAYS) -> int:
    """Borra jobs terminados más antiguos que `days`. Los documentos subidos no
    deben quedarse en el servidor indefinidamente."""
    if days <= 0:
        return 0
    corte = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id FROM jobs WHERE status IN (?, ?) AND created_at < ?",
            (STATUS_DONE, STATUS_ERROR, corte),
        ).fetchall()
    for row in rows:
        delete_job(row["id"])
    return len(rows)
