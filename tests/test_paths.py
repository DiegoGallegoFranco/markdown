"""pipeline/paths.py: rutas de entrada/salida."""
import os
from pathlib import Path

from pipeline import paths as P


def test_slugify():
    assert P.slugify("Informe final (v2)") == "Informe_final_-v2-"
    assert P.slugify("sin_cambios") == "sin_cambios"


def test_common_root(tmp_path):
    a = tmp_path / "x" / "a.pdf"
    b = tmp_path / "x" / "y" / "b.pdf"
    assert P.common_root([]) == Path.cwd()
    assert P.common_root([a]) == (tmp_path / "x").resolve()
    assert P.common_root([a, b]) == (tmp_path / "x").resolve()


def test_relative_key_dentro_y_fuera_del_root(tmp_path):
    root = tmp_path / "docs"
    dentro = root / "sub" / "a.pdf"
    assert P.relative_key(dentro, root) == Path("sub/a.pdf")
    # Fuera del root: no lanza, espeja la ruta absoluta sin el ancla.
    fuera = tmp_path / "otro" / "b.pdf"
    rel = P.relative_key(fuera, root)
    assert not rel.is_absolute()
    assert rel == Path(*fuera.resolve().parts[1:])


def test_doc_out_dir_sanea_carpetas_y_nombre(tmp_path):
    root = tmp_path / "in"
    src = root / "Área (2024)" / "Mi informe (final).pdf"
    out = P.doc_out_dir(src, tmp_path / "out", root)
    assert out == tmp_path / "out" / "Área_-2024-" / "Mi_informe_-final-"


def test_doc_out_dir_fuera_del_root_no_genera_doble_barra(tmp_path):
    out = P.doc_out_dir("/otro/volumen/doc.pdf", tmp_path / "out", tmp_path / "in")
    assert "//" not in str(out)
    assert str(out).startswith(str(tmp_path / "out"))


def test_collect_documents(tmp_path):
    (tmp_path / "sub").mkdir()
    for n in ["a.pdf", "B.PDF", "c.docx", "d.pptx", "e.txt", "~$a.docx", "sub/f.pdf"]:
        (tmp_path / n).write_bytes(b"x")
    nombres = [p.relative_to(tmp_path).as_posix() for p in P.collect_documents(tmp_path)]
    # Ignora los temporales de Office (~$) y extensiones no soportadas.
    assert nombres == sorted(["B.PDF", "a.pdf", "c.docx", "d.pptx", "sub/f.pdf"])
    # Un archivo suelto: sí si su extensión vale, nada si no.
    assert P.collect_documents(tmp_path / "a.pdf") == [tmp_path / "a.pdf"]
    assert P.collect_documents(tmp_path / "e.txt") == []
    assert P.collect_documents(tmp_path, exts={".pdf"}) == sorted(
        [tmp_path / "B.PDF", tmp_path / "a.pdf", tmp_path / "sub" / "f.pdf"])
