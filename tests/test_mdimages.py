"""pipeline/mdimages.py: la única definición de "referencia a imagen"."""
from pathlib import Path

import pytest

from pipeline import mdimages as M

# Lo que emite pandoc para una imagen con atributos: HTML partido en varias líneas.
PANDOC_MULTILINEA = (
    'Texto previo\n\n'
    '<img src="media/image1.png"\n'
    '     style="width:2.5in;height:1in" />\n\n'
    'Más texto\n\n'
    '<img\n'
    'src="media/image2.png" />\n'
)


def test_referencias_markdown():
    texto = "![alt](assets/a.png) y ![](assets/b.jpg \"título\") y ![x](<assets/c.png>)"
    assert M.image_refs(texto) == ["assets/a.png", "assets/b.jpg", "assets/c.png"]


def test_referencias_html_multilinea_de_pandoc():
    assert M.image_refs(PANDOC_MULTILINEA) == ["media/image1.png", "media/image2.png"]


def test_img_con_comillas_simples_y_mayusculas():
    assert M.image_refs("<IMG alt='x' SRC='media/x.png'>") == ["media/x.png"]


def test_mezcla_de_formatos():
    texto = "![](assets/a.png)\n\n" + PANDOC_MULTILINEA
    assert sorted(M.image_refs(texto)) == ["assets/a.png", "media/image1.png", "media/image2.png"]


@pytest.mark.parametrize("ref, externa", [
    ("http://ej.com/a.png", True),
    ("https://ej.com/a.png", True),
    ("data:image/png;base64,AAAA", True),
    ("assets/a.png", False),
    ("media/image1.png", False),
    ("/abs/a.png", False),
])
def test_is_external(ref, externa):
    assert M.is_external(ref) is externa


def test_local_image_refs_descarta_externas():
    texto = ("![](https://cdn.ej.com/logo.png)\n\n![](assets/a.png)\n\n"
             '<img src="http://x.org/b.png" />\n\n<img src="media/c.png" />')
    assert M.local_image_refs(texto) == ["assets/a.png", "media/c.png"]


def test_clean_ref_y_resolve():
    assert M.clean_ref("assets/mi%20img.png?v=2#ancla") == "assets/mi img.png"
    md = Path("/tmp/doc/doc.md")
    assert M.resolve(md, "assets/mi%20img.png") == Path("/tmp/doc/assets/mi img.png")
    assert M.ref_name("media/sub/x%20y.png#a") == "x y.png"


def test_reference_pattern_captura_etiqueta_html_completa():
    m = M.reference_pattern("image1.png").search(PANDOC_MULTILINEA)
    assert m is not None
    # Incluye el cierre `/>` que está una línea más abajo: el caption se
    # inserta DETRÁS de la etiqueta entera, no a mitad.
    assert m.group(0).startswith("<img") and m.group(0).endswith("/>")
    assert "style=" in m.group(0)


def test_reference_pattern_markdown_y_escape():
    texto = "![a](assets/x.png) ![b](assets/x+png)"
    m = M.reference_pattern("x+png").search(texto)
    # El nombre se escapa: `+` no es un cuantificador.
    assert m.group(0) == "![b](assets/x+png)"
    assert M.reference_pattern("x.png").search(texto).group(0) == "![a](assets/x.png)"
    assert M.reference_pattern("otra.png").search(texto) is None


def test_reference_pattern_no_casa_nombres_que_solo_terminan_igual():
    texto = "![](assets/ba.png)\n\n![](assets/a.png)"
    assert M.reference_pattern("a.png").search(texto).group(0) == "![](assets/a.png)"
