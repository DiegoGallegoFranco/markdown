"""pipeline/converters.py: pptx, pymupdf4llm, Marker (con dobles) y la sesión
de llama-server.

Marker y surya NO se instalan: `marker_single` y `llama-server` son scripts
falsos en tests/dobles/, instalados al frente del PATH por la fixture
`binarios`, que además mata en el teardown cualquier proceso que dejen vivo.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time

import pytest

import fabricas
from pipeline import converters as cv


def _pdf_falso(directorio, nombre="doc.pdf"):
    directorio.mkdir(parents=True, exist_ok=True)
    p = directorio / nombre
    p.write_bytes(b"%PDF-1.4 falso")
    return p


# --------------------------------------------------------------------------- despacho

def test_engine_for():
    assert cv.engine_for("a.PDF", "marker") == "marker"
    assert cv.engine_for("a.docx") == "pandoc"
    assert cv.engine_for("a.pptx") == "pptx"
    with pytest.raises(ValueError):
        cv.engine_for("a.txt")


def test_convert_batch_sigue_tras_formato_no_soportado(tmp_path, monkeypatch):
    vistos = []

    def falso(src, out_root, input_root, skip_existing=False, log=None):
        vistos.append(src.name)
        return {"ok": True, "src": str(src)}

    monkeypatch.setitem(cv.CONVERTERS, "pymupdf", falso)
    entrada = tmp_path / "in"
    entrada.mkdir()
    a, raro, b = entrada / "a.pdf", entrada / "x.txt", entrada / "b.pdf"
    for p in (a, raro, b):
        p.write_bytes(b"x")
    res = cv.convert_batch([a, raro, b], tmp_path / "out")
    assert vistos == ["a.pdf", "b.pdf"]
    assert [r["ok"] for r in res] == [True, False, True]
    assert "Formato no soportado" in res[1]["error"]


# --------------------------------------------------------------------------- pptx

def _pptx(path):
    pptx = pytest.importorskip("pptx")
    from pptx.util import Inches

    prs = pptx.Presentation()
    s1 = prs.slides.add_slide(prs.slide_layouts[5])   # solo título
    s1.shapes.title.text = "Título uno"
    caja = s1.shapes.add_textbox(Inches(1), Inches(2), Inches(4), Inches(1))
    caja.text_frame.text = "Primer punto"
    caja.text_frame.add_paragraph().text = "Segundo punto"
    img = fabricas.png(path.parent / "img.png", 80, 60)
    s1.shapes.add_picture(str(img), Inches(1), Inches(4))
    s1.notes_slide.notes_text_frame.text = "Nota del orador"

    s2 = prs.slides.add_slide(prs.slide_layouts[6])   # en blanco
    tabla = s2.shapes.add_table(2, 2, Inches(1), Inches(1), Inches(4), Inches(1)).table
    tabla.cell(0, 0).text, tabla.cell(0, 1).text = "Col|A", "Col B"
    tabla.cell(1, 0).text, tabla.cell(1, 1).text = "x", "línea1\nlínea2"
    grupo = s2.shapes.add_group_shape()
    grupo.shapes.add_textbox(Inches(1), Inches(3), Inches(2), Inches(1)).text_frame.text = "Dentro del grupo"
    prs.save(path)
    return path


def test_convert_pptx(tmp_path):
    entrada = tmp_path / "in"
    entrada.mkdir()
    src = _pptx(entrada / "Mi presentación (v1).pptx")
    r = cv.convert_pptx(src, tmp_path / "out", entrada)
    assert r["ok"], r["error"]
    out_dir = tmp_path / "out" / "Mi_presentación_-v1-"
    md = out_dir / "Mi_presentación_-v1-.md"
    assert r["md"] == str(md) and r["imagenes"] == 1 and r["engine"] == "pptx"
    texto = md.read_text(encoding="utf-8")
    assert "## Diapositiva 1" in texto and "## Diapositiva 2" in texto
    assert "**Título uno**" in texto
    assert "- Primer punto\n- Segundo punto" in texto
    assert "![](assets/slide001-001.png)" in texto
    assert (out_dir / "assets" / "slide001-001.png").exists()
    assert "> Notas: Nota del orador" in texto
    assert "| Col\\|A | Col B |" in texto            # | escapado
    assert "| x | línea1<br>línea2 |" in texto        # salto de línea en celda
    assert "- Dentro del grupo" in texto             # formas dentro de grupos

    # skip_existing: no reconvierte.
    r2 = cv.convert_pptx(src, tmp_path / "out", entrada, skip_existing=True)
    assert r2["skipped"] is True and r2["imagenes"] == 1


def test_convert_pptx_corrupto_devuelve_error(tmp_path):
    pytest.importorskip("pptx")
    src = tmp_path / "roto.pptx"
    src.write_bytes(b"no es un zip")
    r = cv.convert_pptx(src, tmp_path / "out", tmp_path)
    assert r["ok"] is False and r["error"]


# --------------------------------------------------------------------------- pymupdf4llm

def test_convert_pymupdf(tmp_path):
    pytest.importorskip("pymupdf4llm")
    import pymupdf

    entrada = tmp_path / "in docs"
    entrada.mkdir()
    doc = pymupdf.open()
    pag = doc.new_page()
    pag.insert_text((72, 72), "Informe de prueba", fontsize=24)
    pag.insert_textbox(pymupdf.Rect(72, 100, 540, 300), fabricas.LOREM, fontsize=11)
    img = fabricas.png(tmp_path / "img.png", 300, 200, 90)
    pag.insert_image(pymupdf.Rect(72, 320, 372, 520), filename=str(img))
    src = entrada / "mi doc (1).pdf"
    doc.save(src)
    doc.close()

    r = cv.convert_pymupdf(src, tmp_path / "out", entrada)
    assert r["ok"], r["error"]
    out_dir = tmp_path / "out" / "mi_doc_-1-"
    md = out_dir / "mi_doc_-1-.md"
    assert r["md"] == str(md)
    texto = md.read_text(encoding="utf-8")
    assert "Lorem ipsum" in texto
    from pipeline.mdimages import local_image_refs, resolve
    refs = local_image_refs(texto)
    assert refs and r["imagenes"] == len(list((out_dir / "assets").glob("*")))
    for ref in refs:
        # Enlaces relativos al .md (no al CWD), y apuntan a archivos reales.
        assert ref.startswith("assets/")
        assert resolve(md, ref).exists()

    r2 = cv.convert_pymupdf(src, tmp_path / "out", entrada, skip_existing=True)
    assert r2["skipped"] is True


def test_convert_pymupdf_corrupto_devuelve_error(tmp_path):
    pytest.importorskip("pymupdf4llm")
    src = tmp_path / "roto.pdf"
    src.write_bytes(b"basura")
    r = cv.convert_pymupdf(src, tmp_path / "out", tmp_path)
    assert r["ok"] is False and r["error"]


# --------------------------------------------------------------------------- Marker (marker_single falso)

def test_marker_progreso_limitado_imagenes_y_aplanado(tmp_path, binarios, monkeypatch):
    binarios.instalar("marker_ok.py", "marker_single")
    # Con un intervalo enorme solo se emiten: la primera barra y el último
    # estado pendiente (100%). Sin el límite serían 21 líneas.
    monkeypatch.setattr(cv, "_MARKER_PROGRESS_EVERY_S", 1000)
    entrada = tmp_path / "in"
    src = _pdf_falso(entrada, "Mi doc (v2).pdf")
    log = []
    r = cv.convert_marker(src, tmp_path / "out", entrada, log=log.append)
    assert r["ok"], r["error"]

    out_dir = tmp_path / "out" / "Mi_doc_-v2-"
    # Aplanado: la subcarpeta sin sanear `Mi doc (v2)/` desaparece.
    assert not (out_dir / "Mi doc (v2)").exists()
    assert (out_dir / "Mi doc (v2).md").exists()
    assert (out_dir / "_page_0_Picture_1.jpeg").exists()
    assert (out_dir / "Mi doc (v2)_meta.json").exists()
    # Conteo de imágenes sin el _meta.json.
    assert r["imagenes"] == 1

    barras = [x for x in log if "%|" in x]
    assert len(barras) == 2
    assert "0%|" in barras[0] and "100%|" in barras[-1]
    assert all("\r" not in x for x in log)
    assert any(x.endswith("marker: Loading models") for x in log)
    assert any(x.endswith("marker: Saved markdown") for x in log)

    # skip_existing tras el aplanado.
    r2 = cv.convert_marker(src, tmp_path / "out", entrada, skip_existing=True)
    assert r2["skipped"] is True and r2["imagenes"] == 1


def test_marker_mode_se_pasa_si_esta_fijado(tmp_path, binarios, monkeypatch):
    binarios.instalar("marker_ok.py", "marker_single")
    env_log = tmp_path / "env.jsonl"
    monkeypatch.setenv("FAKE_ENV_LOG", str(env_log))
    entrada = tmp_path / "in"
    src = _pdf_falso(entrada)
    cv.convert_marker(src, tmp_path / "o1", entrada)
    monkeypatch.setattr(cv, "MARKER_MODE", "balanced")
    cv.convert_marker(src, tmp_path / "o2", entrada)
    argvs = [json.loads(x)["argv"] for x in env_log.read_text().splitlines()]
    assert "--mode" not in argvs[0]
    assert argvs[1][-2:] == ["--mode", "balanced"]


def test_marker_error_con_ultimas_lineas(tmp_path, binarios):
    binarios.instalar("marker_falla.py", "marker_single")
    entrada = tmp_path / "in"
    src = _pdf_falso(entrada)
    log = []
    r = cv.convert_marker(src, tmp_path / "out", entrada, log=log.append)
    assert r["ok"] is False
    err = r["error"]
    assert err.startswith("marker_single rc=3:")
    assert "RuntimeError: boom" in err and "linea 9" in err
    assert "linea 4" not in err   # solo las últimas 5 líneas
    assert "%|" not in err        # las barras de progreso no cuentan como error
    assert any("ERROR marker doc.pdf: rc=3" in x for x in log)


def test_marker_no_instalado(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "vacio"))
    entrada = tmp_path / "in"
    r = cv.convert_marker(_pdf_falso(entrada), tmp_path / "out", entrada)
    assert r["ok"] is False and "no está en el PATH" in r["error"]


def test_marker_sin_md_es_error(tmp_path, binarios):
    script = binarios.dir / "marker_single"
    script.write_text(f"#!{sys.executable}\nprint('nada')\n")
    script.chmod(0o755)
    entrada = tmp_path / "in"
    r = cv.convert_marker(_pdf_falso(entrada), tmp_path / "out", entrada)
    assert r["ok"] is False and "sin producir" in r["error"]


def test_marker_timeout_mata_al_grupo_incluido_hijo(tmp_path, binarios, monkeypatch, util):
    binarios.instalar("marker_cuelga.py", "marker_single")
    pids = tmp_path / "pids.txt"
    monkeypatch.setenv("HANG_PIDS", str(pids))
    monkeypatch.setattr(cv, "MARKER_TIMEOUT_S", 1)
    entrada = tmp_path / "in"
    t0 = time.monotonic()
    r = cv.convert_marker(_pdf_falso(entrada), tmp_path / "out", entrada)
    assert time.monotonic() - t0 < 10
    assert r["ok"] is False and r["error"] == "timeout tras 1s"
    pid_marker, pid_hijo = map(int, pids.read_text().split())
    # El hijo ignora SIGTERM: solo cae con el SIGKILL al grupo.
    assert util.esperar(lambda: not util.pid_vivo(pid_hijo), timeout=5)
    assert not util.pid_vivo(pid_marker)


def test_marker_nieto_con_el_pipe_heredado_no_cuelga(tmp_path, binarios, monkeypatch):
    binarios.instalar("marker_hereda.py", "marker_single")
    monkeypatch.setenv("INH_PID", str(tmp_path / "nieto.pid"))
    entrada = tmp_path / "in"
    t0 = time.monotonic()
    log = []
    r = cv.convert_marker(_pdf_falso(entrada), tmp_path / "out", entrada, log=log.append)
    assert time.monotonic() - t0 < 8
    assert r["ok"], r["error"]
    assert any("marker: hecho" in x for x in log)
    # El nieto sigue vivo (no es del grupo de marker); lo mata el teardown.


# --------------------------------------------------------------------------- marker_session + llama-server falso

def test_backend_llamacpp_por_defecto_y_respetado(monkeypatch):
    monkeypatch.delenv("SURYA_INFERENCE_BACKEND", raising=False)
    monkeypatch.delenv("SURYA_INFERENCE_KEEP_ALIVE", raising=False)
    env = cv._marker_env()
    assert env["SURYA_INFERENCE_BACKEND"] == "llamacpp"
    assert "SURYA_INFERENCE_KEEP_ALIVE" not in env
    monkeypatch.setenv("SURYA_INFERENCE_BACKEND", "vllm")
    assert cv._marker_env()["SURYA_INFERENCE_BACKEND"] == "vllm"
    with cv.marker_session():
        assert cv._marker_env()["SURYA_INFERENCE_KEEP_ALIVE"] == "true"
    assert "SURYA_INFERENCE_KEEP_ALIVE" not in cv._marker_env()


def _docs(tmp_path, n=3):
    entrada = tmp_path / "in"
    return entrada, [_pdf_falso(entrada, f"doc{i}.pdf") for i in range(n)]


def test_sin_sesion_cada_documento_arranca_su_servidor(tmp_path, surya_falso):
    entrada, docs = _docs(tmp_path)
    for d in docs:
        assert cv.convert(d, tmp_path / "out", entrada, pdf_engine="marker")["ok"]
    assert len(surya_falso.lanzados()) == 3
    assert cv._llama_server_pid() is None
    entornos = [json.loads(x) for x in surya_falso.env_log.read_text().splitlines()]
    assert all(e["backend"] == "llamacpp" and e["keep_alive"] is None for e in entornos)


def test_lote_arranca_el_servidor_una_vez_y_lo_apaga(tmp_path, surya_falso, util):
    entrada, docs = _docs(tmp_path)
    log = []
    res = cv.convert_batch(docs, tmp_path / "out", entrada, pdf_engine="marker", log=log.append)
    assert all(r["ok"] for r in res)
    lanzados = surya_falso.lanzados()
    assert len(lanzados) == 1
    assert util.esperar(lambda: not util.pid_vivo(lanzados[0]))
    assert not surya_falso.sentinel.exists()
    assert any("llama-server) detenido" in x for x in log)
    entornos = [json.loads(x) for x in surya_falso.env_log.read_text().splitlines()]
    assert all(e["keep_alive"] == "true" for e in entornos)
    assert (tmp_path / "out" / "doc0" / "doc0.md").exists()


def test_servidor_preexistente_no_se_apaga(tmp_path, surya_falso):
    ajeno = subprocess.Popen([str(surya_falso.bin.dir / "llama-server")],
                             start_new_session=True, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    try:
        surya_falso.sentinel.write_text(json.dumps({"pid": ajeno.pid, "port": 1}))
        assert cv._llama_server_pid() == ajeno.pid
        entrada, docs = _docs(tmp_path)
        res = cv.convert_batch(docs, tmp_path / "out", entrada, pdf_engine="marker")
        assert all(r["ok"] for r in res)
        assert surya_falso.lanzados() == []       # se reutilizó
        assert ajeno.poll() is None               # y sigue vivo
        assert surya_falso.sentinel.exists()
    finally:
        ajeno.kill()
        ajeno.wait()


def test_sesiones_solapadas_apagan_al_cerrar_la_ultima(tmp_path, surya_falso, util):
    entrada, docs = _docs(tmp_path, 2)
    dentro, salir, errores = threading.Event(), threading.Event(), []

    def job_a():
        try:
            with cv.marker_session():
                assert cv.convert(docs[0], tmp_path / "out", entrada, pdf_engine="marker")["ok"]
                dentro.set()
                salir.wait(15)
        except BaseException as e:  # noqa: BLE001
            errores.append(e)
            dentro.set()

    hilo = threading.Thread(target=job_a)
    hilo.start()
    try:
        assert dentro.wait(15)
        with cv.marker_session():
            assert cv.convert(docs[1], tmp_path / "out", entrada, pdf_engine="marker")["ok"]
        lanzados = surya_falso.lanzados()
        assert len(lanzados) == 1                  # compartido entre las dos
        assert cv._is_llama_server(lanzados[0])    # A sigue abierta: vivo
    finally:
        salir.set()
        hilo.join(15)
    assert not errores
    assert util.esperar(lambda: not util.pid_vivo(lanzados[0]))
    assert cv._session_depth == 0


def test_pid_ajeno_en_el_sentinel_no_se_mata(tmp_path, surya_falso):
    otro = subprocess.Popen(["sleep", "30"])
    try:
        surya_falso.sentinel.write_text(json.dumps({"pid": otro.pid, "port": 1}))
        assert cv._llama_server_pid() is None
        cv._stop_llama_server()
        with cv.marker_session():
            pass
        assert otro.poll() is None
    finally:
        otro.kill()
        otro.wait()


def test_sentinel_ilegible(surya_falso):
    surya_falso.sentinel.write_text("{no es json")
    assert cv._llama_server_pid() is None
    surya_falso.sentinel.write_text(json.dumps({"port": 1}))
    assert cv._llama_server_pid() is None
