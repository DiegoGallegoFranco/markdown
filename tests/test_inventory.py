"""pipeline/inventory.py: clasificación nativo / escaneado / mixto.

Los PDFs son sintéticos (tests/fabricas.py): una página "escaneada" es una
página con una sola imagen rasterizada que la cubre entera, sin capa de texto.
"""
from __future__ import annotations

import csv
import importlib
import os

import pytest

import fabricas
from pipeline import inventory

VARS = ("DOC2MD_COBERTURA_ESCANEO", "DOC2MD_MIXTO_MIN_PAGINAS", "DOC2MD_MIXTO_MIN_RATIO")


@pytest.fixture
def recargar():
    """Las DOC2MD_* se leen al importar: se fijan y se recarga el módulo. Al
    terminar se restaura el entorno y se recarga de nuevo con los defaults."""
    originales = {v: os.environ.get(v) for v in VARS}

    def cargar(**env):
        for k, v in env.items():
            os.environ[k] = str(v)
        return importlib.reload(inventory)

    yield cargar
    for k, v in originales.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    importlib.reload(inventory)


def analizar(path, mod=None):
    mod = mod or inventory
    return mod.analyze_pdf(path, path.parent)


def test_defaults():
    assert inventory.COBERTURA_ESCANEO == 0.5
    assert inventory.MIXTO_MIN_PAGINAS == 2
    assert inventory.MIXTO_MIN_RATIO == 0.10


def test_nativo(tmp_path):
    r = analizar(fabricas.pdf(tmp_path / "nativo.pdf", "n" * 6))
    assert r["tipo"] == "nativo" and r["motor_sugerido"] == "pymupdf"
    assert r["paginas"] == 6 and r["paginas_con_texto"] == 6 and r["ratio_texto"] == 1.0
    assert r["paginas_escaneadas"] == 0 and r["motivo"] == ""
    assert r["ruta"] == "nativo.pdf"


def test_escaneado(tmp_path):
    r = analizar(fabricas.pdf(tmp_path / "esc.pdf", "e" * 4))
    assert r["tipo"] == "escaneado" and r["motor_sugerido"] == "marker"
    assert r["paginas_escaneadas"] == 4 and r["lista_escaneadas"] == "1,2,3,4"
    assert r["imagenes_embebidas"] == 1  # misma imagen compartida: dedup por xref
    assert "0/4" in r["motivo"]


def test_mixto(tmp_path):
    r = analizar(fabricas.pdf(tmp_path / "mixto.pdf", "n" * 10 + "e" * 3))
    assert r["tipo"] == "mixto" and r["motor_sugerido"] == "marker"
    assert r["paginas_escaneadas"] == 3 and r["lista_escaneadas"] == "11,12,13"
    assert "pymupdf4llm las dejaría vacías" in r["motivo"]


def test_documento_corto_una_pagina_escaneada_pasa_el_ratio(tmp_path):
    # 1/9 = 11% >= 10%: en documentos cortos una sola página basta.
    r = analizar(fabricas.pdf(tmp_path / "corto.pdf", "n" * 8 + "e"))
    assert r["tipo"] == "mixto"


def test_paginas_blancas_no_diluyen_el_ratio(tmp_path):
    r = analizar(fabricas.pdf(tmp_path / "blancas.pdf", "nb" * 6))
    assert r["tipo"] == "nativo"
    assert r["paginas_en_blanco"] == 6
    assert r["ratio_texto"] == 0.5  # la columna se sigue reportando sobre el total


def test_portada_unica_bajo_umbral_sigue_nativo(tmp_path):
    # 1 página-imagen en 30 (3%, < 2 páginas): decorativa, se avisa en motivo.
    r = analizar(fabricas.pdf(tmp_path / "portada.pdf", "e" + "n" * 29))
    assert r["tipo"] == "nativo" and r["motor_sugerido"] == "pymupdf"
    assert r["paginas_escaneadas"] == 1 and r["lista_escaneadas"] == "1"
    assert "ignorada(s) por umbral" in r["motivo"]


def test_lista_escaneadas_se_trunca(tmp_path):
    r = analizar(fabricas.pdf(tmp_path / "largo.pdf", "e" * 22))
    assert r["lista_escaneadas"] == ",".join(str(i) for i in range(1, 21)) + ",…"


def test_env_mixto_min_paginas(tmp_path, recargar):
    pdf = fabricas.pdf(tmp_path / "portada.pdf", "e" + "n" * 29)
    mod = recargar(DOC2MD_MIXTO_MIN_PAGINAS=1)
    assert mod.MIXTO_MIN_PAGINAS == 1
    assert analizar(pdf, mod)["tipo"] == "mixto"


def test_env_mixto_min_ratio(tmp_path, recargar):
    pdf = fabricas.pdf(tmp_path / "corto.pdf", "n" * 8 + "e")
    mod = recargar(DOC2MD_MIXTO_MIN_RATIO=0.5)
    assert analizar(pdf, mod)["tipo"] == "nativo"


def test_env_cobertura_escaneo(tmp_path, recargar):
    import pymupdf
    doc = pymupdf.open()
    for i in range(6):
        fabricas.pagina_nativa(doc, i)
    for _ in range(3):
        fabricas.pagina_escaneada(doc, cobertura=0.8)  # imagen al 80% de la página
    pdf = tmp_path / "parcial.pdf"
    doc.save(pdf)
    doc.close()
    assert analizar(pdf)["tipo"] == "mixto"            # 0.8 >= 0.5
    mod = recargar(DOC2MD_COBERTURA_ESCANEO=0.95)
    r = analizar(pdf, mod)
    assert r["paginas_escaneadas"] == 0                 # 0.8 < 0.95


def test_env_invalida_usa_default(recargar):
    mod = recargar(DOC2MD_COBERTURA_ESCANEO="abc", DOC2MD_MIXTO_MIN_PAGINAS="dos",
                   DOC2MD_MIXTO_MIN_RATIO="")
    assert mod.COBERTURA_ESCANEO == 0.5
    assert mod.MIXTO_MIN_PAGINAS == 2
    assert mod.MIXTO_MIN_RATIO == 0.10


def test_corrupto_no_tumba_el_inventario(tmp_path):
    malo = tmp_path / "corrupto.pdf"
    malo.write_bytes(b"%PDF-1.4\nbasura no es un pdf\n%%EOF")
    bueno = fabricas.pdf(tmp_path / "bueno.pdf", "nn")
    mensajes = []
    filas = inventory.build_inventory([malo, bueno], tmp_path, on_progress=mensajes.append)
    assert filas[0]["tipo"] == "ERROR" and filas[0]["motor_sugerido"] == "marker"
    assert filas[0]["motivo"].startswith("no se pudo analizar")
    assert filas[1]["tipo"] == "nativo"
    assert any(m.startswith("ERROR analizando") for m in mensajes)
    salida = inventory.write_csv(filas, tmp_path / "out" / "inventario.csv")
    with open(salida, encoding="utf-8") as f:
        lector = csv.DictReader(f)
        assert lector.fieldnames == inventory.FIELDNAMES
        assert [r["tipo"] for r in lector] == ["ERROR", "nativo"]


def test_progreso_explica_los_mixtos(tmp_path):
    pdf = fabricas.pdf(tmp_path / "mixto.pdf", "n" * 10 + "e" * 3)
    mensajes = []
    inventory.build_inventory([pdf], tmp_path, on_progress=mensajes.append)
    assert "mixto (3 página(s)" in mensajes[-1]


def test_ruta_fuera_del_root(tmp_path):
    pdf = fabricas.pdf(tmp_path / "a" / "x.pdf", "n")
    r = inventory.analyze_pdf(pdf, tmp_path / "otro")
    assert r["ruta"] == str(pdf)
