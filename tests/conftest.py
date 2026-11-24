"""Fixtures compartidas de la suite.

Varios módulos leen variables de entorno AL IMPORTARSE (webapp/jobs.py:
DATA_DIR, WORKERS...; pipeline/captions.py: OLLAMA_HOST, CAPTION_*;
pipeline/converters.py: MARKER_*; pipeline/inventory.py: DOC2MD_*). Por eso:

1. Aquí arriba, ANTES de que ningún test importe esos módulos, se fija un
   entorno neutro: nada del .env ni del shell del desarrollador se cuela, y
   DATA_DIR apunta a un directorio temporal (nunca a ./data del repo).
2. Cada test que toca estado de módulo lo aísla con monkeypatch de los
   atributos (DATA_DIR, DB_PATH, el sentinel de llama-server...), no con env.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

# No ensuciar el repo con __pycache__ de los módulos importados por los tests.
sys.dont_write_bytecode = True

# --------------------------------------------------------------------------- entorno al importar

_DATA_DIR_SESION = Path(tempfile.mkdtemp(prefix="doc2md-tests-"))

for _var in (
    "CAPTION_BACKEND", "CAPTION_MODEL", "CAPTION_WORKERS", "CAPTION_MAX_RETRIES",
    "CAPTION_MIN_PX", "CAPTION_MAX_PX", "ANTHROPIC_API_KEY", "OLLAMA_TIMEOUT_S",
    "MARKER_MODE", "MARKER_TIMEOUT_S", "SURYA_INFERENCE_BACKEND",
    "SURYA_INFERENCE_KEEP_ALIVE", "DOC2MD_COBERTURA_ESCANEO",
    "DOC2MD_MIXTO_MIN_PAGINAS", "DOC2MD_MIXTO_MIN_RATIO", "PURGE_INTERVAL_H",
    "MAX_UPLOAD_MB",
):
    os.environ.pop(_var, None)
os.environ["DATA_DIR"] = str(_DATA_DIR_SESION)
os.environ["WORKERS"] = "1"
os.environ["RETENTION_DAYS"] = "14"
# Puerto 9 (discard): nada escucha. Si algún test se olvidara de simular
# Ollama, falla rápido con "connection refused" en vez de llamar a uno real.
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:9"

DOBLES = Path(__file__).parent / "dobles"


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_DATA_DIR_SESION, ignore_errors=True)


# --------------------------------------------------------------------------- llama-server: nunca el real

@pytest.fixture(autouse=True)
def sentinel_aislado(tmp_path, monkeypatch):
    """Redirige el sentinel de surya a tmp_path en TODOS los tests.

    marker_session()/convert_batch() pueden llamar a _stop_llama_server(), que
    mata el PID que diga el sentinel: sin esto, un test podría apagar el
    llama-server real del desarrollador (~/.cache/datalab/...).
    """
    from pipeline import converters
    sentinel = tmp_path / "surya" / "llamacpp_server.json"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(converters, "_SURYA_LLAMA_SENTINEL", sentinel)
    monkeypatch.setattr(converters, "_session_depth", 0)
    monkeypatch.setattr(converters, "_session_preexisting", False)
    # Caché de extracciones de imágenes: nunca la real (~/.cache o DATA_DIR).
    from pipeline import captions
    monkeypatch.setattr(captions, "CAPTION_CACHE_DIR", tmp_path / "caption-cache")
    yield sentinel


# --------------------------------------------------------------------------- binarios falsos en el PATH

def _matar_procesos_con(marca: str) -> None:
    """SIGKILL a todo proceso cuya línea de comandos contenga `marca`.

    Los dobles se instalan en un directorio único por test, y los procesos que
    lanzan (llama-server falso, hijos colgados) llevan esa ruta en su argv, así
    que esto limpia exactamente lo de este test y nada más.
    """
    try:
        out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return
    for linea in out.splitlines():
        pid_txt, _, cmd = linea.strip().partition(" ")
        if marca in cmd and pid_txt.isdigit() and int(pid_txt) != os.getpid():
            try:
                os.kill(int(pid_txt), signal.SIGKILL)
            except OSError:
                pass


@pytest.fixture
def binarios(tmp_path, monkeypatch):
    """Instala dobles de prueba (scripts Python ejecutables) al frente del PATH.

    Uso: `binarios.instalar("marker_ok.py", "marker_single")`. En el teardown
    mata cualquier proceso que siga vivo lanzado desde ese directorio, aunque
    el test haya fallado a mitad.
    """
    bindir = tmp_path / "bin_falsos"
    bindir.mkdir()
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")

    def instalar(fuente: str, nombre: str) -> Path:
        destino = bindir / nombre
        codigo = (DOBLES / fuente).read_text(encoding="utf-8")
        destino.write_text(f"#!{sys.executable}\n{codigo}", encoding="utf-8")
        destino.chmod(0o755)
        return destino

    ns = SimpleNamespace(dir=bindir, instalar=instalar)
    yield ns
    _matar_procesos_con(str(bindir))


@pytest.fixture
def surya_falso(binarios, sentinel_aislado, tmp_path, monkeypatch):
    """marker_single + llama-server falsos que imitan a surya 0.22.1.

    marker_single se engancha al llama-server del sentinel si está vivo, o
    arranca uno (registrando su PID en `spawns`), y lo mata al salir salvo
    SURYA_INFERENCE_KEEP_ALIVE=true. Todo con stdout/stderr del servidor a
    DEVNULL, como el real (que escribe a un log).
    """
    binarios.instalar("marker_ok.py", "marker_single")
    binarios.instalar("llama_server.py", "llama-server")
    spawns = tmp_path / "spawns.txt"
    env_log = tmp_path / "marker_env.jsonl"
    monkeypatch.setenv("FAKE_SENTINEL", str(sentinel_aislado))
    monkeypatch.setenv("FAKE_SPAWNS", str(spawns))
    monkeypatch.setenv("FAKE_ENV_LOG", str(env_log))

    def lanzados() -> list[int]:
        return [int(x) for x in spawns.read_text().split()] if spawns.exists() else []

    return SimpleNamespace(bin=binarios, sentinel=sentinel_aislado,
                           lanzados=lanzados, env_log=env_log)


def esperar(cond, timeout=5.0, paso=0.05) -> bool:
    """Sondea `cond()` hasta que sea cierta o venza el plazo."""
    fin = time.monotonic() + timeout
    while time.monotonic() < fin:
        if cond():
            return True
        time.sleep(paso)
    return bool(cond())


def pid_vivo(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # Un zombi sigue respondiendo a kill(0); para estos tests cuenta como muerto.
    try:
        out = subprocess.run(["ps", "-p", str(pid), "-o", "stat="],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return True
    return bool(out.strip()) and not out.strip().startswith("Z")


@pytest.fixture
def util():
    return SimpleNamespace(esperar=esperar, pid_vivo=pid_vivo)


# --------------------------------------------------------------------------- web: almacén aislado

@pytest.fixture
def entorno_web(tmp_path, monkeypatch):
    """webapp.jobs apuntando a un DATA_DIR propio del test, con cola y workers
    nuevos, y el estado de captions simulado (nada de consultar Ollama)."""
    import queue
    import threading

    from pipeline import captions as captions_mod
    from webapp import jobs, main

    data = tmp_path / "data"
    monkeypatch.setattr(jobs, "DATA_DIR", data)
    monkeypatch.setattr(jobs, "DB_PATH", data / "jobs.db")
    monkeypatch.setattr(jobs, "JOBS_DIR", data / "jobs")
    monkeypatch.setattr(jobs, "WORKERS", 1)
    monkeypatch.setattr(jobs, "RETENTION_DAYS", 14)
    # Cola y registro de workers nuevos: los hilos de otros tests quedan
    # bloqueados (daemon) en SU cola vieja y no tocan este almacén.
    monkeypatch.setattr(jobs, "_queue", queue.Queue())
    monkeypatch.setattr(jobs, "_workers_started", threading.Event())
    monkeypatch.setattr(jobs, "_worker_threads", [])
    monkeypatch.setattr(main, "_captions_status", None)
    monkeypatch.setattr(captions_mod, "backend_disponible",
                        lambda *a, **k: (False, "simulado en tests"))
    jobs.init_db()
    yield SimpleNamespace(jobs=jobs, main=main, data=data)
    # Si un test arrancó el purgador, se detiene antes de deshacer los parches.
    jobs.stop_purger()
    if jobs._purge_thread is not None:
        jobs._purge_thread.join(5)
        jobs._purge_thread = None


@pytest.fixture
def cliente(entorno_web):
    """TestClient sin lifespan (no arranca workers ni purga) y sin seguir
    redirecciones, para poder comprobar los 303."""
    from fastapi.testclient import TestClient
    return TestClient(entorno_web.main.app, follow_redirects=False)
