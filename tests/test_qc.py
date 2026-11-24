"""pipeline/qc.py: control de calidad del Markdown generado."""
import csv

import fabricas
from pipeline import qc


def _md(path, texto):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(texto, encoding="utf-8")
    return path


def test_estado_ok_con_imagenes_y_captions(tmp_path):
    d = tmp_path / "out" / "doc"
    fabricas.png(d / "assets" / "a.png", 10, 10)
    fabricas.png(d / "media" / "b.png", 10, 10, 50)
    md = _md(d / "doc.md",
             "# Título\n\n![](assets/a.png)\n\n*[descripción generada por IA: algo]*\n\n"
             '<img src="media/b.png"\n style="w" />\n\n![](https://ej.com/x.png)\n')
    fila = qc.check_md(md, tmp_path / "out", duracion_s=90)
    assert fila["estado"] == "OK"
    assert fila["imagenes"] == 2          # la externa no cuenta
    assert fila["captions"] == 1
    assert fila["enlaces_rotos"] == 0
    assert fila["encabezados"] == 1
    assert fila["duracion_min"] == 1.5
    assert fila["ruta_md"] == "doc/doc.md"


def test_estado_vacio(tmp_path):
    md = _md(tmp_path / "doc.md", "  \n\n ")
    assert qc.check_md(md, tmp_path)["estado"] == "ERROR: vacío"


def test_estado_enlaces_rotos_tiene_prioridad_sobre_encabezados(tmp_path):
    md = _md(tmp_path / "doc.md", "texto\n\n![](assets/no.png)\n\n<img src=\"media/tampoco.png\">")
    fila = qc.check_md(md, tmp_path)
    assert fila["enlaces_rotos"] == 2
    assert fila["estado"] == "REVISION MANUAL: 2 enlace(s) roto(s)"


def test_estado_sin_encabezados(tmp_path):
    md = _md(tmp_path / "doc.md", "solo texto, sin títulos")
    assert qc.check_md(md, tmp_path)["estado"] == "REVISION MANUAL: sin encabezados"


def test_duracion_cero_es_dato_valido(tmp_path):
    md = _md(tmp_path / "doc.md", "# x")
    assert qc.check_md(md, tmp_path, duracion_s=0.0)["duracion_min"] == 0.0
    assert qc.check_md(md, tmp_path)["duracion_min"] == ""


def test_run_summarize_y_csv(tmp_path):
    out = tmp_path / "output"
    _md(out / "a" / "a.md", "# A\n\n*[descripción generada por IA: x]*")
    _md(out / "b" / "b.md", "")
    _md(out / "c" / "c.md", "sin titulo")
    filas = qc.run(out, root=out, durations={"a": 60})
    assert [f["ruta_md"] for f in filas] == ["a/a.md", "b/b.md", "c/c.md"]
    assert filas[0]["duracion_min"] == 1.0
    resumen = qc.summarize(filas)
    assert resumen == {"documentos": 3, "ok": 1, "revision": 1, "error": 1,
                       "imagenes": 0, "captions": 1}
    csv_path = qc.write_csv(filas, tmp_path / "x" / "final.csv")
    with open(csv_path, encoding="utf-8") as f:
        assert list(csv.DictReader(f))[0]["estado"] == "OK"
