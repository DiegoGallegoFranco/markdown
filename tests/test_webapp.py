"""webapp/jobs.py + webapp/main.py: cola, worker, rutas, subida y purga.

Cada test trabaja sobre un DATA_DIR propio (fixture `entorno_web`), con cola y
workers nuevos, y sin consultar Ollama (backend_disponible simulado).
"""
from __future__ import annotations

import asyncio
import json
import shutil
import threading
import time
import zipfile
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from pipeline import captions as captions_mod


def _esperar_estado(jobs, job_id, estados, util, timeout=10):
    assert util.esperar(lambda: (jobs.get(job_id) or {}).get("status") in estados, timeout), \
        f"job {job_id} en {jobs.get(job_id)}"
    return jobs.get(job_id)


def _antiguo(dias):
    return (datetime.now(timezone.utc) - timedelta(days=dias)).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- worker

def test_worker_sobrevive_a_una_excepcion(entorno_web, monkeypatch, util):
    jobs = entorno_web.jobs
    procesados = []

    def run_que_explota(job_id):
        if job_id == "bomba":
            raise RuntimeError("boom fuera del try")
        procesados.append(job_id)

    monkeypatch.setattr(jobs, "_run_job", run_que_explota)
    jobs.start_workers()
    jobs.start_workers()  # idempotente: no arranca un segundo pool
    assert len(jobs._worker_threads) == 1
    jobs.enqueue("bomba")
    jobs.enqueue("siguiente")
    assert util.esperar(lambda: procesados == ["siguiente"])
    assert jobs.workers_alive() == jobs.WORKERS == 1


def test_job_con_carpeta_borrada_en_cola_termina_en_error(entorno_web, util):
    jobs = entorno_web.jobs
    jobs.start_workers()
    borrado = jobs.create_job("auto", False)
    shutil.rmtree(jobs.job_dir(borrado))          # fila existe, carpeta no
    jobs.enqueue(borrado)
    job = _esperar_estado(jobs, borrado, {"error"}, util)
    assert "no se subió ningún documento" in job["error"]
    assert jobs.workers_alive() == 1

    # El hilo sigue procesando: el siguiente job también se atiende.
    otro = jobs.create_job("auto", False)
    (jobs.job_dir(otro) / "input" / "nota.txt").write_text("x")
    jobs.enqueue(otro)
    _esperar_estado(jobs, otro, {"error"}, util)
    assert "ERROR FATAL" in jobs.read_log(otro)
    assert jobs.workers_alive() == 1


def test_run_job_inexistente_no_hace_nada(entorno_web):
    entorno_web.jobs._run_job("no-existe")  # no lanza


def test_requeue_orphans(entorno_web):
    jobs = entorno_web.jobs
    corriendo = jobs.create_job("auto", False)
    jobs.update(corriendo, status=jobs.STATUS_RUNNING)
    en_cola = jobs.create_job("auto", False)
    hecho = jobs.create_job("auto", False)
    jobs.update(hecho, status=jobs.STATUS_DONE)
    assert jobs.requeue_orphans() == 2
    assert jobs.get(corriendo)["status"] == jobs.STATUS_QUEUED
    assert {jobs._queue.get_nowait(), jobs._queue.get_nowait()} == {corriendo, en_cola}
    assert "Reanudando" in jobs.read_log(corriendo)


# --------------------------------------------------------------------------- _run_job con Marker falso

@pytest.fixture
def job_marker(entorno_web, binarios):
    """Un job de un PDF con motor marker (marker_single falso) y captions pedidas."""
    binarios.instalar("marker_ok.py", "marker_single")
    jobs = entorno_web.jobs
    jid = jobs.create_job("marker", True)
    (jobs.job_dir(jid) / "input" / "doc.pdf").write_bytes(b"%PDF-1.4")
    return jid


def test_si_captions_falla_el_job_termina_done_con_zip(entorno_web, job_marker, monkeypatch):
    jobs = entorno_web.jobs
    monkeypatch.setattr(captions_mod, "backend_disponible", lambda *a, **k: (True, "falso"))

    def run_roto(*a, **k):
        raise ConnectionError("Ollama se cayó a mitad")

    monkeypatch.setattr(captions_mod, "run", run_roto)
    jobs._run_job(job_marker)
    job = jobs.get(job_marker)
    assert job["status"] == "done", job["error"]
    assert job["summary"]["captions_error"] == "ConnectionError: Ollama se cayó a mitad"
    assert job["summary"]["convertidos"] == 1 and job["summary"]["fallidos"] == 0
    zip_path = jobs.job_dir(job_marker) / "resultado.zip"
    with zipfile.ZipFile(zip_path) as z:
        nombres = z.namelist()
    assert "doc/doc.md" in nombres and "inventario_final.csv" in nombres
    log = jobs.read_log(job_marker)
    assert "AVISO: el captioning falló" in log
    assert "marker: Saved markdown" in log

    # La página del job muestra el aviso.
    from fastapi.testclient import TestClient
    r = TestClient(entorno_web.main.app).get(f"/jobs/{job_marker}")
    assert r.status_code == 200
    assert "captioning falló" in r.text and "Ollama se cayó a mitad" in r.text


def test_captions_no_disponible_se_omite_sin_error(entorno_web, job_marker, monkeypatch):
    jobs = entorno_web.jobs
    llamado = []
    monkeypatch.setattr(captions_mod, "run", lambda *a, **k: llamado.append(1))
    jobs._run_job(job_marker)
    job = jobs.get(job_marker)
    assert job["status"] == "done"
    assert "captions_error" not in job["summary"]
    assert llamado == []
    assert "Paso omitido" in jobs.read_log(job_marker)


# --------------------------------------------------------------------------- rutas

def test_healthz_ok_y_503_sin_workers(entorno_web, cliente, monkeypatch):
    jobs = entorno_web.jobs
    vivo = type("HiloVivo", (), {"is_alive": lambda self: True})()
    muerto = type("HiloMuerto", (), {"is_alive": lambda self: False})()
    monkeypatch.setattr(jobs, "_worker_threads", [vivo])
    r = cliente.get("/healthz")
    assert r.status_code == 200 and r.text == "ok"
    monkeypatch.setattr(jobs, "_worker_threads", [muerto])
    r = cliente.get("/healthz")
    assert r.status_code == 503 and "0/1" in r.text
    monkeypatch.setattr(jobs, "_worker_threads", [])
    assert cliente.get("/healthz").status_code == 503


def test_portada_no_bloquea_el_bucle_y_cachea(entorno_web, monkeypatch):
    main = entorno_web.main
    llamadas = []
    lento = 1.0

    def backend_lento(*a, **k):
        llamadas.append(time.monotonic())
        time.sleep(lento)
        return (False, "simulado lento")

    monkeypatch.setattr(captions_mod, "backend_disponible", backend_lento)
    monkeypatch.setattr(entorno_web.jobs, "_worker_threads",
                        [type("H", (), {"is_alive": lambda self: True})()])

    async def escenario():
        transporte = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transporte, base_url="http://test") as c:
            async def health():
                await asyncio.sleep(0.2)   # la portada ya está esperando a "Ollama"
                t0 = time.monotonic()
                r = await c.get("/healthz")
                return r.status_code, time.monotonic() - t0

            portada, (codigo, dt) = await asyncio.gather(c.get("/"), health())
            t0 = time.monotonic()
            segunda = await c.get("/")
            return portada, codigo, dt, segunda, time.monotonic() - t0

    portada, codigo, dt, segunda, dt2 = asyncio.run(escenario())
    assert portada.status_code == 200 and "simulado lento" in portada.text
    assert codigo == 200 and dt < 0.5, f"/healthz esperó {dt:.2f}s detrás de la portada"
    assert segunda.status_code == 200 and dt2 < 0.5
    assert len(llamadas) == 1                         # la segunda usó la caché

    # Con el TTL vencido se vuelve a consultar.
    monkeypatch.setattr(main, "CAPTIONS_STATUS_TTL_S", 0)
    lento = 0
    assert main.captions_status() == (False, "simulado lento")
    assert len(llamadas) == 2


def test_captions_status_no_tumba_la_portada(entorno_web, cliente, monkeypatch):
    def roto(*a, **k):
        raise RuntimeError("inesperado")

    monkeypatch.setattr(captions_mod, "backend_disponible", roto)
    r = cliente.get("/")
    assert r.status_code == 200
    assert entorno_web.main.captions_status()[0] is False


def test_borrar_job_running_409_y_terminado_303(entorno_web, cliente):
    jobs = entorno_web.jobs
    jid = jobs.create_job("auto", False)
    jobs.update(jid, status=jobs.STATUS_RUNNING)
    r = cliente.post(f"/jobs/{jid}/delete")
    assert r.status_code == 409
    assert jobs.get(jid) is not None and jobs.job_dir(jid).exists()
    jobs.update(jid, status=jobs.STATUS_DONE)
    r = cliente.post(f"/jobs/{jid}/delete")
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert jobs.get(jid) is None and not jobs.job_dir(jid).exists()


def test_rutas_de_consulta(entorno_web, cliente):
    jobs = entorno_web.jobs
    assert cliente.get("/jobs/nada").status_code == 404
    assert cliente.get("/api/jobs/nada").status_code == 404
    assert cliente.get("/jobs/nada/log").status_code == 404
    assert cliente.get("/jobs/nada/download").status_code == 404
    jid = jobs.create_job("auto", False)
    assert cliente.get(f"/jobs/{jid}/download").status_code == 409
    assert cliente.get(f"/api/jobs/{jid}").json()["status"] == "queued"
    assert cliente.get("/api/jobs").json()["jobs"][0]["id"] == jid
    (jobs.job_dir(jid) / "resultado.zip").write_bytes(b"PK")
    r = cliente.get(f"/jobs/{jid}/download")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"


# --------------------------------------------------------------------------- subida

@pytest.fixture
def encolados(entorno_web, monkeypatch):
    lista = []
    monkeypatch.setattr(entorno_web.jobs, "enqueue", lista.append)
    return lista


def test_subida_renombra_colisiones_e_ignora_extensiones(entorno_web, cliente, encolados):
    jobs = entorno_web.jobs
    archivos = [
        ("files", ("informe final.pdf", b"%PDF-1", "application/pdf")),
        ("files", ("informe_final.pdf", b"%PDF-2", "application/pdf")),  # mismo slug
        ("files", ("A.pdf", b"%PDF-3", "application/pdf")),
        ("files", ("a.pdf", b"%PDF-4", "application/pdf")),              # mayúsculas
        ("files", ("a.pdf", b"%PDF-5", "application/pdf")),              # repetido
        ("files", ("notas.txt", b"hola", "text/plain")),                 # no soportado
        ("files", ("../../escape.pdf", b"%PDF-6", "application/pdf")),   # ruta maliciosa
    ]
    r = cliente.post("/jobs", data={"engine": "pymupdf"}, files=archivos)
    assert r.status_code == 303
    jid = r.headers["location"].rsplit("/", 1)[-1]
    assert encolados == [jid]
    entrada = jobs.job_dir(jid) / "input"
    contenido = {p.name: p.read_bytes() for p in entrada.iterdir()}
    assert contenido == {
        "informe final.pdf": b"%PDF-1",
        "informe_final-2.pdf": b"%PDF-2",
        "A.pdf": b"%PDF-3",
        "a-2.pdf": b"%PDF-4",
        "a-3.pdf": b"%PDF-5",
        "escape.pdf": b"%PDF-6",
    }
    job = jobs.get(jid)
    assert job["n_files"] == 6 and job["engine"] == "pymupdf" and job["captions"] is False


def test_nombre_unico():
    from webapp.main import _nombre_unico
    usados = set()
    assert _nombre_unico("Informe Final.pdf", usados) == "Informe Final.pdf"
    assert _nombre_unico("informe_final.docx", usados) == "informe_final-2.docx"
    assert _nombre_unico("INFORME FINAL.pdf", usados) == "INFORME FINAL-3.pdf"
    assert _nombre_unico("otro.pdf", usados) == "otro.pdf"


def test_subida_solo_no_soportados_400_y_sin_job(entorno_web, cliente, encolados):
    r = cliente.post("/jobs", data={"engine": "auto"},
                     files=[("files", ("a.txt", b"x", "text/plain"))])
    assert r.status_code == 400
    assert entorno_web.jobs.list_jobs() == [] and encolados == []
    assert list((entorno_web.data / "jobs").iterdir()) == []


def test_subida_motor_invalido_400(entorno_web, cliente, encolados):
    r = cliente.post("/jobs", data={"engine": "rm -rf"},
                     files=[("files", ("a.pdf", b"x", "application/pdf"))])
    assert r.status_code == 400 and entorno_web.jobs.list_jobs() == []


def test_subida_supera_max_upload_413(entorno_web, cliente, encolados, monkeypatch):
    monkeypatch.setattr(entorno_web.jobs, "MAX_UPLOAD_MB", 1)
    grande = b"0" * (1024 * 1024 + 10)
    r = cliente.post("/jobs", data={"engine": "auto"}, files=[
        ("files", ("ok.pdf", b"%PDF", "application/pdf")),
        ("files", ("grande.pdf", grande, "application/pdf")),
    ])
    assert r.status_code == 413 and "grande.pdf" in r.text
    # Nada a medias: ni fila, ni carpeta, ni job encolado.
    assert entorno_web.jobs.list_jobs() == [] and encolados == []
    assert list((entorno_web.data / "jobs").iterdir()) == []


def test_subida_con_captions_sin_backend_desactiva_captions(entorno_web, cliente, encolados):
    r = cliente.post("/jobs", data={"engine": "auto", "captions": "on"},
                     files=[("files", ("a.pdf", b"%PDF", "application/pdf"))])
    jid = r.headers["location"].rsplit("/", 1)[-1]
    assert entorno_web.jobs.get(jid)["captions"] is False


def test_subida_con_captions_y_backend_listo(entorno_web, cliente, encolados, monkeypatch):
    monkeypatch.setattr(captions_mod, "backend_disponible", lambda *a, **k: (True, "ok"))
    r = cliente.post("/jobs", data={"engine": "auto", "captions": "on"},
                     files=[("files", ("a.pdf", b"%PDF", "application/pdf"))])
    jid = r.headers["location"].rsplit("/", 1)[-1]
    assert entorno_web.jobs.get(jid)["captions"] is True


# --------------------------------------------------------------------------- purga

def test_purge_old_solo_borra_terminados_antiguos(entorno_web):
    jobs = entorno_web.jobs
    viejo_done = jobs.create_job("auto", False)
    jobs.update(viejo_done, status=jobs.STATUS_DONE, created_at=_antiguo(30))
    viejo_error = jobs.create_job("auto", False)
    jobs.update(viejo_error, status=jobs.STATUS_ERROR, created_at=_antiguo(30))
    viejo_running = jobs.create_job("auto", False)
    jobs.update(viejo_running, status=jobs.STATUS_RUNNING, created_at=_antiguo(30))
    viejo_queued = jobs.create_job("auto", False)
    jobs.update(viejo_queued, status=jobs.STATUS_QUEUED, created_at=_antiguo(30))
    reciente = jobs.create_job("auto", False)
    jobs.update(reciente, status=jobs.STATUS_DONE)

    assert jobs.purge_old(0) == 0          # 0 = nunca
    assert jobs.purge_old(14) == 2
    assert jobs.get(viejo_done) is None and not jobs.job_dir(viejo_done).exists()
    assert jobs.get(viejo_error) is None
    for jid in (viejo_running, viejo_queued, reciente):
        assert jobs.get(jid) is not None and jobs.job_dir(jid).exists()


def test_purgador_periodico_sobrevive_y_se_detiene(entorno_web, monkeypatch, util, capsys):
    jobs = entorno_web.jobs
    viejo = jobs.create_job("auto", False)
    jobs.update(viejo, status=jobs.STATUS_DONE, created_at=_antiguo(30))
    viejo_run = jobs.create_job("auto", False)
    jobs.update(viejo_run, status=jobs.STATUS_RUNNING, created_at=_antiguo(30))

    jobs.start_purger(interval_h=0.1 / 3600)          # cada 0,1 s
    hilo = jobs._purge_thread
    jobs.start_purger(interval_h=0.1 / 3600)          # idempotente
    assert jobs._purge_thread is hilo
    assert util.esperar(lambda: jobs.get(viejo) is None)
    assert jobs.get(viejo_run) is not None

    # Una excepción en una pasada no mata el hilo: sigue intentándolo.
    intentos = []

    def purga_rota(*a, **k):
        intentos.append(1)
        raise RuntimeError("disco lleno")

    monkeypatch.setattr(jobs, "purge_old", purga_rota)
    assert util.esperar(lambda: len(intentos) >= 3)
    assert hilo.is_alive()
    assert "[purga] error" in capsys.readouterr().out

    jobs.stop_purger()
    hilo.join(2)
    assert not hilo.is_alive()


def test_purgador_desactivado(entorno_web, monkeypatch):
    jobs = entorno_web.jobs
    jobs.start_purger(interval_h=0)
    assert jobs._purge_thread is None
    monkeypatch.setattr(jobs, "RETENTION_DAYS", 0)
    jobs.start_purger(interval_h=1)
    assert jobs._purge_thread is None


# --------------------------------------------------------------------------- arranque (lifespan)

def test_lifespan_arranca_workers_purga_y_reanuda(entorno_web, monkeypatch, util):
    from fastapi.testclient import TestClient
    jobs = entorno_web.jobs
    procesados = []
    monkeypatch.setattr(jobs, "_run_job", procesados.append)
    huerfano = jobs.create_job("auto", False)
    jobs.update(huerfano, status=jobs.STATUS_RUNNING)
    caducado = jobs.create_job("auto", False)
    jobs.update(caducado, status=jobs.STATUS_DONE, created_at=_antiguo(30))

    with TestClient(entorno_web.main.app) as c:
        assert c.get("/healthz").status_code == 200
        assert util.esperar(lambda: procesados == [huerfano])
        assert jobs.get(caducado) is None
        assert jobs._purge_thread is not None and jobs._purge_thread.is_alive()
    jobs._purge_thread.join(2)
    assert not jobs._purge_thread.is_alive()       # stop_purger al cerrar
