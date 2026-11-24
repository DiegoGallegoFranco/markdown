"""Modo detallado: extracción estructurada por tipo de imagen, bloque en el .md,
imagenes.json por documento, índice global, caché y rehacer.

El modelo se simula con `_ollama_post`, que devuelve el JSON que respondería
qwen3-vl según la imagen (se reconoce por su color de relleno).
"""
from __future__ import annotations

import base64
import csv
import json
import threading

import pymupdf
import pytest

import fabricas
from pipeline import captions as C
from pipeline import extraccion as X
from pipeline import indice, qc
from pipeline.mdimages import contar_descripciones, es_descripcion_ia, image_refs

TABLA = {
    "tipo": "tabla", "titulo": "Presupuesto por dependencia 2025",
    "resumen": "Compara el presupuesto de 2024 y 2025 por dependencia.",
    "contenido": "| Dependencia | 2024 | 2025 |\n|---|---|---|\n| Logística | 1.250 | 1.410 |\n"
                 "| Personal | 3.100 | 3.050 |",
    "texto_visible": "Fuente: Oficina de Planeación",
    "palabras_clave": ["presupuesto", "dependencias", "2025"],
    "entidades": ["Logística", "Personal"], "calidad": "alta",
}
DIAGRAMA = {
    "tipo": "diagrama", "titulo": "Organigrama", "resumen": "Estructura jerárquica.",
    "contenido": "Dirección → Logística\nDirección → Personal", "texto_visible": "",
    "palabras_clave": ["organigrama"], "entidades": ["Dirección"], "calidad": "media",
}
LOGO = {"tipo": "logo", "titulo": "Escudo institucional", "resumen": "", "contenido": "",
        "texto_visible": "", "palabras_clave": [], "entidades": [], "calidad": "alta"}
# Color de relleno de cada imagen sintética -> respuesta del modelo
RESPUESTAS = {50: LOGO, 200: TABLA, 77: DIAGRAMA}


@pytest.fixture
def modelo(monkeypatch):
    monkeypatch.setattr(C, "CAPTION_MODE", C.MODO_DETALLADO)
    monkeypatch.setattr(C, "CAPTION_MIN_PX", 64)
    monkeypatch.setattr(C, "CAPTION_MAX_PX", 1600)
    llamadas, lock = [], threading.Lock()

    def post(path, payload, host=None, timeout=None):
        raw = base64.b64decode(payload["messages"][0]["images"][0])
        pix = pymupdf.Pixmap(raw)
        valor = pix.samples[0]
        with lock:
            llamadas.append({"payload": payload, "valor": valor, "w": pix.width})
        return {"message": {"content": json.dumps(RESPUESTAS.get(valor, TABLA))}}

    monkeypatch.setattr(C, "_ollama_post", post)
    monkeypatch.setattr(C, "check_ollama", lambda *a, **k: None)
    return llamadas


def _documento(raiz):
    d = raiz / "informe"
    for i in range(3):
        fabricas.png(d / "assets" / f"informe.pdf-{i + 1:04d}-01.png", 200, 100, 50)  # logo x3
    fabricas.png(d / "assets" / "informe.pdf-0002-02.png", 400, 300, 200)   # tabla, pág. 2
    fabricas.png(d / "assets" / "informe.pdf-0003-02.png", 300, 300, 77)    # diagrama, pág. 3
    fabricas.png(d / "assets" / "informe.pdf-0003-03.png", 16, 16, 10)      # diminuta
    (d / "informe.md").write_text(
        "# Informe anual\n\n![](assets/informe.pdf-0001-01.png)\n\n"
        "## 3. Finanzas\n\n### 3.2 Presupuesto\n\n"
        "El cuadro siguiente resume la ejecución por dependencia.\n\n"
        "![](assets/informe.pdf-0002-01.png)\n\n![](assets/informe.pdf-0002-02.png)\n\n"
        "Tabla 4. Presupuesto 2024-2025\n\n"
        "## 4. Organización\n\n![](assets/informe.pdf-0003-01.png)\n\n"
        "![](assets/informe.pdf-0003-02.png)\n\n![](assets/informe.pdf-0003-03.png)\n",
        encoding="utf-8")
    return d


# --------------------------------------------------------------------------- unidades

def test_pagina_de_cada_conversor():
    assert X.pagina_de("informe.pdf-0012-01.png") == (12, "página")   # pymupdf4llm, 1-based
    assert X.pagina_de("_page_0_Picture_3.jpeg") == (1, "página")      # marker, 0-based
    assert X.pagina_de("slide004-002.png") == (4, "diapositiva")
    assert X.pagina_de("image3.png") == (None, "página")               # pandoc


def test_contexto_seccion_pie_y_numero():
    md = ("# A\n\n![](assets/x.png)\n\n## B\n\n### B.1\n\nPárrafo previo.\n\n"
          "![](assets/y.png)\n\nFigura 2. Mapa de la zona\n")
    ctx = X.contexto(md, "y.png")
    assert ctx["numero"] == 2
    assert ctx["seccion"] == "A > B > B.1"
    assert ctx["antes"] == "Párrafo previo."
    assert ctx["pie"].startswith("Figura 2.")
    assert "Pie de figura: Figura 2." in X.prompt_con_contexto(ctx)


def test_interpretar_json_y_respuesta_rota():
    datos, ok = X.interpretar("basura {\"tipo\": \"GRÁFICA\", \"titulo\": \"T\"} basura")
    assert ok and datos["tipo"] == "grafica" and datos["calidad"] == "media"
    datos, ok = X.interpretar('{"tipo": "tabla", "contenido": "| a | b |\n| 1')  # cortada
    assert not ok and datos["calidad"] == "baja" and "| a | b |" in datos["contenido"]


def test_render_bloque_es_un_solo_parrafo_y_no_parece_imagen():
    datos = X.normalizar({**TABLA, "contenido": TABLA["contenido"] + "\n\n![truco](x.png)"})
    b = X.render_bloque(datos, 7, "informe/img-007", 12, "página", "3. Finanzas")
    assert "\n\n" not in b and all(l.startswith(">") for l in b.splitlines())
    assert es_descripcion_ia(b) and image_refs(b) == []
    assert b.startswith("> **Imagen 7 · Tabla** — Presupuesto por dependencia 2025")
    assert "id: informe/img-007 · página 12 · sección: 3. Finanzas" in b
    assert "> | Logística | 1.250 | 1.410 |" in b
    logo = X.render_bloque(X.normalizar(LOGO), 1, "d/img-001", None, "página", "")
    assert len(logo.splitlines()) == 2


# --------------------------------------------------------------------------- run() detallado

def test_run_detallado_bloques_sidecar_e_indice(tmp_path, modelo):
    d = _documento(tmp_path)
    r = C.run(tmp_path)
    # 3 logos idénticos -> 1 llamada; + tabla + diagrama; la diminuta no se envía
    assert len(modelo) == 3 and r["llamadas"] == 3 and r["ok"] == 5
    assert r["omitidas_pequenas"] == 1 and r["modo"] == "detallado"

    # La petición lleva esquema, ventana de contexto y el contexto del documento
    p_tabla = next(c["payload"] for c in modelo if c["valor"] == 200)
    assert p_tabla["format"] == X.ESQUEMA and p_tabla["options"]["num_ctx"] == C.CAPTION_NUM_CTX
    prompt = p_tabla["messages"][0]["content"]
    assert "Informe anual > 3. Finanzas > 3.2 Presupuesto" in prompt
    assert "Tabla 4. Presupuesto 2024-2025" in prompt

    texto = (d / "informe.md").read_text(encoding="utf-8")
    assert contar_descripciones(texto) == 5
    assert "> **Imagen 3 · Tabla** — Presupuesto por dependencia 2025" in texto
    assert "id: informe/img-003 · página 2 · sección: Informe anual > 3. Finanzas > 3.2 Presupuesto" in texto
    assert "> | Personal | 3.100 | 3.050 |" in texto
    assert "> Dirección → Logística" in texto

    side = json.loads((d / "imagenes.json").read_text(encoding="utf-8"))
    assert [e["numero"] for e in side] == [1, 2, 3, 4, 5]
    tabla = next(e for e in side if e["tipo"] == "tabla")
    assert tabla["archivo"] == "assets/informe.pdf-0002-02.png" and tabla["pagina"] == 2
    assert tabla["contenido"].startswith("| Dependencia")
    reusadas = [e for e in side if e["reutilizada_de"]]
    assert len(reusadas) == 2 and all(e["tipo"] == "logo" for e in reusadas)

    # Índice global
    md_ind = (tmp_path / "INDICE_IMAGENES.md").read_text(encoding="utf-8")
    assert "5 imágenes en 1 documento(s)" in md_ind
    assert "## Tabla" in md_ind and "## Diagrama" in md_ind and "## Logo" not in md_ind
    assert "[informe/img-003](<informe/assets/informe.pdf-0002-02.png>)" in md_ind
    filas = list(csv.DictReader(open(tmp_path / "indice_imagenes.csv", encoding="utf-8")))
    assert len(filas) == 5 and filas[0]["tipo"] == "tabla"
    assert filas[0]["palabras_clave"] == "presupuesto; dependencias; 2025"
    todo = json.loads((tmp_path / "indice_imagenes.json").read_text(encoding="utf-8"))
    assert todo[0]["contenido"].startswith("| Dependencia")


def test_run_detallado_reanudar_y_cache_entre_trabajos(tmp_path, modelo):
    _documento(tmp_path / "trabajo1")
    C.run(tmp_path / "trabajo1")
    antes = len(modelo)
    C.run(tmp_path / "trabajo1")                      # reanudar: nada pendiente
    assert len(modelo) == antes
    d2 = _documento(tmp_path / "trabajo2")           # mismas imágenes, otro trabajo
    r = C.run(tmp_path / "trabajo2")
    assert len(modelo) == antes, "la caché por contenido debía evitar el modelo"
    assert r["desde_cache"] == 3 and r["ok"] == 5
    assert contar_descripciones((d2 / "informe.md").read_text(encoding="utf-8")) == 5


def test_respuesta_rota_no_se_cachea(tmp_path, modelo, monkeypatch):
    _documento(tmp_path)
    monkeypatch.setattr(C, "_ollama_post", lambda *a, **k: {"message": {"content": "{rota"}})
    r = C.run(tmp_path)
    assert r["extraccion_incompleta"] == 3 and r["ok"] == 5
    assert not any((C.CAPTION_CACHE_DIR or tmp_path / "nada").glob("*.json"))


def test_qc_cuenta_detallados_e_ignora_el_indice(tmp_path, modelo):
    _documento(tmp_path)
    C.run(tmp_path)
    filas = qc.run(tmp_path, root=tmp_path)
    assert [f["ruta_md"] for f in filas] == ["informe/informe.md"]
    assert filas[0]["captions"] == 5 and filas[0]["imagenes"] == 6


def test_dedup_reconoce_bloques_detallados(tmp_path, modelo):
    d = _documento(tmp_path)
    C.run(tmp_path)
    md = d / "informe.md"
    texto = md.read_text(encoding="utf-8")
    bloque = next(p for p in texto.split("\n\n") if p.startswith("> **Imagen 3"))
    md.write_text(texto.replace(bloque, bloque + "\n\n" + bloque), encoding="utf-8")
    assert C.dedup(tmp_path)["duplicados_eliminados"] == 1
    assert md.read_text(encoding="utf-8") == texto


def test_rehacer_pasa_de_breve_a_detallado(tmp_path, modelo):
    d = _documento(tmp_path)
    md = d / "informe.md"
    md.write_text(md.read_text(encoding="utf-8").replace(
        "![](assets/informe.pdf-0002-02.png)",
        "![](assets/informe.pdf-0002-02.png)\n\n*[descripción generada por IA: una tabla]*"),
        encoding="utf-8")
    assert C.quitar_descripciones(tmp_path)["quitadas"] == 1
    C.run(tmp_path)
    texto = md.read_text(encoding="utf-8")
    assert "descripción generada por IA: una tabla" not in texto
    assert "> **Imagen 3 · Tabla**" in texto
    # rehacer-todo quita también los detallados y su imagenes.json
    r = C.quitar_descripciones(tmp_path, solo_breves=False)
    assert r["quitadas"] == 5 and not (d / "imagenes.json").exists()


def test_modo_breve_sigue_disponible(tmp_path, modelo, monkeypatch):
    d = _documento(tmp_path)
    monkeypatch.setattr(C, "_ollama_post",
                        lambda *a, **k: {"message": {"content": "descripción corta"}})
    r = C.run(tmp_path, modo="breve")
    assert r["modo"] == "breve" and r["ok"] == 5
    texto = (d / "informe.md").read_text(encoding="utf-8")
    assert texto.count("*[descripción generada por IA: descripción corta]*") == 5
    assert not (d / "imagenes.json").exists()


def test_indice_vacio_y_regenerable(tmp_path):
    r = indice.construir(tmp_path)
    assert r["imagenes"] == 0 and (tmp_path / "INDICE_IMAGENES.md").exists()


def test_variables_vacias_usan_el_defecto(monkeypatch):
    # docker-compose pasa `${VAR:-}` como cadena vacía: no debe tumbar el import.
    monkeypatch.setenv("CAPTION_MAX_PX", "")
    assert C._env_int("CAPTION_MAX_PX", "1600") == 1600
    monkeypatch.setenv("CAPTION_MAX_PX", "900")
    assert C._env_int("CAPTION_MAX_PX", "1600") == 900


def test_indice_md_no_lista_logos_pero_csv_si(tmp_path, modelo):
    _documento(tmp_path)
    C.run(tmp_path)
    md_ind = (tmp_path / "INDICE_IMAGENES.md").read_text(encoding="utf-8")
    assert "## Logo" not in md_ind and "3 logo(s)/imagen(es) decorativa(s)" in md_ind
    filas = list(csv.DictReader(open(tmp_path / "indice_imagenes.csv", encoding="utf-8")))
    assert sum(f["tipo"] == "logo" for f in filas) == 3
