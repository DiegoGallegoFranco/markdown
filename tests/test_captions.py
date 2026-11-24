"""pipeline/captions.py: estado, inserción, selección de imágenes y run().

El modelo se simula monkeypatcheando `_ollama_post` y `check_ollama`: ninguna
prueba llama a Ollama ni a la API de Anthropic.
"""
from __future__ import annotations

import base64
import hashlib
import threading
from types import SimpleNamespace

import pymupdf
import pytest

import fabricas
from pipeline import captions as C

MARCA = C.MARKER


@pytest.fixture
def modelo(monkeypatch):
    """Modelo de visión falso. Registra cada llamada con el tamaño y el
    media_type REAL de lo que se le envió."""
    monkeypatch.setattr(C, "CAPTION_MIN_PX", 64)
    monkeypatch.setattr(C, "CAPTION_MAX_PX", 1024)
    llamadas = []
    lock = threading.Lock()

    def post(path, payload, host=None, timeout=None):
        raw = base64.b64decode(payload["messages"][0]["images"][0])
        pix = pymupdf.Pixmap(raw)
        with lock:
            llamadas.append(SimpleNamespace(path=path, w=pix.width, h=pix.height,
                                            media=C._sniff_media_type(raw),
                                            model=payload["model"]))
            n = len(llamadas)
        return {"message": {"content": f" desc {pix.width}x{pix.height} #{n} "}}

    monkeypatch.setattr(C, "_ollama_post", post)
    monkeypatch.setattr(C, "check_ollama", lambda *a, **k: None)
    return llamadas


# --------------------------------------------------------------------------- caption_state / insert_caption

def test_caption_state_solo_primera_aparicion(tmp_path):
    md = tmp_path / "doc.md"
    md.write_text(
        "# T\n\n![](assets/a.png)\n\n" + MARCA + " x]*\n\n"
        "![](assets/b.png)\n\ntexto\n\n"
        "![](assets/b.png)\n\n" + MARCA + " repetida]*\n\n"
        '<img src="media/c.png"\n  style="w" />',
        encoding="utf-8")
    assert C.caption_state(md) == {"a.png": True, "b.png": False, "c.png": False}
    assert C.caption_state(tmp_path / "no_existe.md") == {}


def test_insert_caption_idempotente_y_html_multilinea(tmp_path):
    md = tmp_path / "doc.md"
    md.write_text('Texto\n\n<img src="media/image1.png"\n     style="w" />\n\nFin\n',
                  encoding="utf-8")
    assert C.insert_caption(md, "image1.png", "un diagrama") is True
    texto = md.read_text(encoding="utf-8")
    # Detrás de la etiqueta COMPLETA, no a mitad del <img ...>.
    assert 'style="w" />\n\n*[descripción generada por IA: un diagrama]*\n\nFin' in texto
    assert C.insert_caption(md, "image1.png", "otra") is False
    assert md.read_text(encoding="utf-8") == texto
    # Imagen no referenciada: no inserta nada.
    assert C.insert_caption(md, "otra.png", "x") is False


def test_insert_caption_no_confunde_nombres_con_el_mismo_sufijo(tmp_path):
    md = tmp_path / "doc.md"
    md.write_text("![](assets/ba.png)\n\ntexto\n\n![](assets/a.png)\n", encoding="utf-8")
    assert C.insert_caption(md, "a.png", "desc A") is True
    assert C.caption_state(md) == {"ba.png": False, "a.png": True}
    assert C.insert_caption(md, "a.png", "desc A") is False


def test_lock_por_archivo():
    from pathlib import Path
    assert C._lock_for(Path("/a.md")) is C._lock_for(Path("/a.md"))
    assert C._lock_for(Path("/a.md")) is not C._lock_for(Path("/b.md"))


def test_insercion_concurrente_no_pierde_captions(tmp_path):
    """Muchos hilos escribiendo en el MISMO .md: sin el lock por archivo, dos
    hilos leen la misma versión y el segundo write pisa al primero."""
    n = 60
    md = tmp_path / "doc.md"
    md.write_text("# Doc\n\n" + "\n\n".join(f"Pág {i}\n\n![](assets/img{i}.png)" for i in range(n)),
                  encoding="utf-8")
    barrera = threading.Barrier(n)
    resultados = []

    def trabajo(i):
        barrera.wait()
        resultados.append(C.insert_caption(md, f"img{i}.png", f"caption {i}"))

    hilos = [threading.Thread(target=trabajo, args=(i,)) for i in range(n)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join(10)
    texto = md.read_text(encoding="utf-8")
    assert all(resultados) and len(resultados) == n
    assert texto.count(MARCA) == n
    for i in range(n):
        assert f"![](assets/img{i}.png)\n\n*[descripción generada por IA: caption {i}]*" in texto
    assert all(C.caption_state(md).values())


# --------------------------------------------------------------------------- pendientes / diminutas

def _doc_con_imagenes(raiz):
    """docA (assets/, Markdown) y docB (media/, HTML multilínea de pandoc)."""
    a = raiz / "docA"
    for i in range(4):  # el mismo logo en cada página: 4 archivos idénticos
        fabricas.png(a / "assets" / f"logo_p{i}.png", 200, 100, 50)
    fabricas.png(a / "assets" / "icono.png", 16, 16, 10)        # diminuta
    fabricas.png(a / "assets" / "filete.png", 800, 3, 0)        # diminuta (lado menor)
    fabricas.jpg(a / "assets" / "foto.png", 3000, 2000, 128)    # JPEG con extensión .png
    fabricas.png(a / "assets" / "diagrama.png", 400, 300, 200)
    fabricas.png(a / "assets" / "huerfana.png", 300, 300, 99)   # no referenciada
    partes = ["# Doc A"]
    partes += [f"Página {i}\n\n![](assets/logo_p{i}.png)" for i in range(4)]
    partes += ["![](assets/icono.png) viñeta", "![](assets/filete.png)",
               "![foto](assets/foto.png)", "![](assets/diagrama.png)"]
    (a / "docA.md").write_text("\n\n".join(partes) + "\n", encoding="utf-8")

    b = raiz / "docB"
    fabricas.png(b / "media" / "image1.png", 200, 100, 50)   # idéntica al logo de docA
    fabricas.png(b / "media" / "image2.png", 300, 300, 77)
    (b / "docB.md").write_text(
        'Texto\n\n<img src="media/image1.png"\n     style="width:2in" />\n\nMás\n\n'
        '<img\nsrc="media/image2.png" />\n\nFin\n', encoding="utf-8")
    return a, b


def test_pending_images_excluye_diminutas_y_no_referenciadas(tmp_path, modelo):
    _doc_con_imagenes(tmp_path)
    pendientes, total = C.pending_images(tmp_path)
    nombres = sorted(p.name for p in pendientes)
    assert total == 11
    assert nombres == sorted(["logo_p0.png", "logo_p1.png", "logo_p2.png", "logo_p3.png",
                              "foto.png", "diagrama.png", "image1.png", "image2.png"])
    # Las diminutas tampoco aparecen en la auditoría.
    auditoria = C.audit(tmp_path)
    assert len(auditoria) == 8
    assert not any("icono" in x or "filete" in x for x in auditoria)
    # Con el filtro desactivado sí cuentan.
    assert len(C.pending_images(tmp_path, min_px=0)[0]) == 10
    assert len(C.pending_images(tmp_path, limit=3)[0]) == 3


def test_is_tiny(tmp_path):
    chica = fabricas.png(tmp_path / "a.png", 16, 16)
    filete = fabricas.png(tmp_path / "b.png", 800, 3)
    grande = fabricas.png(tmp_path / "c.png", 100, 100)
    ilegible = tmp_path / "d.webp"
    ilegible.write_bytes(b"RIFF\x00\x00\x00\x00WEBPbasura")
    assert C.is_tiny(chica, 64) and C.is_tiny(filete, 64)
    assert not C.is_tiny(grande, 64)
    assert not C.is_tiny(chica, 0)          # 0 desactiva
    assert not C.is_tiny(ilegible, 64)      # ante la duda, se describe


# --------------------------------------------------------------------------- prepare_image / media_type

def test_prepare_image_reduce_en_memoria_sin_tocar_disco(tmp_path):
    foto = fabricas.jpg(tmp_path / "foto.png", 3000, 2000)   # extensión mentirosa
    antes = hashlib.sha256(foto.read_bytes()).hexdigest()
    datos, media = C.prepare_image(foto, max_px=1024)
    pix = pymupdf.Pixmap(datos)
    assert (pix.width, pix.height) == (1024, 683)
    assert media == "image/jpeg" == C._sniff_media_type(datos)
    assert hashlib.sha256(foto.read_bytes()).hexdigest() == antes


def test_prepare_image_sin_cambios_y_media_type_por_bytes(tmp_path):
    p = fabricas.jpg(tmp_path / "mentira.png", 200, 100)
    datos, media = C.prepare_image(p, max_px=1024)
    assert datos == p.read_bytes() and media == "image/jpeg"
    png = fabricas.png(tmp_path / "x.jpg", 50, 50)
    assert C.prepare_image(png, max_px=1024)[1] == "image/png"
    # Grande con transparencia -> PNG reducido.
    alfa = fabricas.png(tmp_path / "alfa.png", 2048, 100, alpha=True)
    datos, media = C.prepare_image(alfa, max_px=1024)
    assert media == "image/png" and pymupdf.Pixmap(datos).width == 1024
    # 0 desactiva la reducción.
    grande = fabricas.png(tmp_path / "g.png", 1500, 10)
    assert C.prepare_image(grande, max_px=0)[0] == grande.read_bytes()


def test_prepare_image_indescifrable_se_envia_tal_cual(tmp_path):
    raro = tmp_path / "x.webp"
    raro.write_bytes(b"RIFF\x00\x00\x00\x00WEBPbasura")
    assert C.prepare_image(raro) == (raro.read_bytes(), "image/webp")


def test_sniff_media_type():
    assert C._sniff_media_type(b"\x89PNG\r\n\x1a\nxxxx") == "image/png"
    assert C._sniff_media_type(b"\xff\xd8\xff\xe0") == "image/jpeg"
    assert C._sniff_media_type(b"GIF89a....") == "image/gif"
    assert C._sniff_media_type(b"RIFF\x00\x00\x00\x00WEBP") == "image/webp"
    assert C._sniff_media_type(b"otra cosa") is None


def test_anthropic_declara_el_media_type_real(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CAPTION_MAX_PX", 1024)
    foto = fabricas.jpg(tmp_path / "foto.png", 100, 100)
    enviado = {}

    class Mensajes:
        def create(self, **kw):
            enviado.update(kw)
            bloque = SimpleNamespace(type="text", text="  una foto ")
            return SimpleNamespace(stop_reason="end_turn", content=[bloque])

    cliente = SimpleNamespace(messages=Mensajes())
    assert C.caption_image(foto, backend="anthropic", client=cliente) == "una foto"
    fuente = enviado["messages"][0]["content"][0]["source"]
    assert fuente["media_type"] == "image/jpeg"


# --------------------------------------------------------------------------- run()

def test_run_dry_run_no_llama_al_modelo(tmp_path, modelo):
    a, _ = _doc_con_imagenes(tmp_path)
    antes = (a / "docA.md").read_text(encoding="utf-8")
    r = C.run(tmp_path, dry_run=True, workers=4)
    assert modelo == []
    assert r["pendientes"] == 8 and r["llamadas"] == 4 and r["omitidas_pequenas"] == 2
    assert r["total"] == 11 and r["ok"] == 0
    assert (a / "docA.md").read_text(encoding="utf-8") == antes


def test_run_completo_duplicados_diminutas_y_reanudar(tmp_path, modelo):
    a, b = _doc_con_imagenes(tmp_path)
    foto_hash = hashlib.sha256((a / "assets" / "foto.png").read_bytes()).hexdigest()
    lineas_log = []
    r = C.run(tmp_path, workers=4, log=lineas_log.append)

    # 5 logos idénticos (4 en docA + 1 en docB) -> 1 sola llamada.
    # Llamadas: logo, foto, diagrama, image2.
    assert len(modelo) == 4
    assert r["llamadas"] == 4
    assert r["ok"] == 8 and r["reutilizadas"] == 4
    assert r["omitidas_pequenas"] == 2 and r["errores"] == 0 and r["sin_referencia"] == 0

    # Nada enviado supera CAPTION_MAX_PX; la foto se mandó reducida y como JPEG.
    assert max(max(c.w, c.h) for c in modelo) <= 1024
    assert any((c.w, c.h, c.media) == (1024, 683, "image/jpeg") for c in modelo)
    assert hashlib.sha256((a / "assets" / "foto.png").read_bytes()).hexdigest() == foto_hash

    ta = (a / "docA.md").read_text(encoding="utf-8")
    tb = (b / "docB.md").read_text(encoding="utf-8")
    logos = [p for p in (ta + "\n\n" + tb).split("\n\n") if p.startswith(MARCA) and "200x100" in p]
    assert len(logos) == 5 and len(set(logos)) == 1
    assert ta.count(MARCA) == 6 and tb.count(MARCA) == 2
    # Las diminutas no dejan rastro en el .md.
    assert "16x16" not in ta and "800x3" not in ta
    assert "![](assets/icono.png) viñeta\n\n![](assets/filete.png)" in ta
    assert C.audit(tmp_path) == []
    assert any("Duplicadas por contenido: 4" in x for x in lineas_log)
    assert any("Omitidas por diminutas" in x for x in lineas_log)

    # Reanudar: nada pendiente, ninguna llamada nueva, .md intactos.
    r2 = C.run(tmp_path, workers=4)
    assert len(modelo) == 4 and r2["pendientes"] == 0 and r2["llamadas"] == 0
    assert (a / "docA.md").read_text(encoding="utf-8") == ta


def test_run_con_lista_de_imagenes_explicita(tmp_path, modelo):
    a, _ = _doc_con_imagenes(tmp_path)
    imgs = [a / "assets" / "diagrama.png", a / "assets" / "icono.png"]
    r = C.run(tmp_path, images=imgs)
    assert r["total"] == 2 and r["pendientes"] == 1 and r["omitidas_pequenas"] == 1
    assert len(modelo) == 1 and r["ok"] == 1


def test_run_un_fallo_no_aborta_el_lote(tmp_path, modelo, monkeypatch):
    a, _ = _doc_con_imagenes(tmp_path)
    post_ok = C._ollama_post

    def post(path, payload, host=None, timeout=None):
        raw = base64.b64decode(payload["messages"][0]["images"][0])
        if pymupdf.Pixmap(raw).width == 400:  # el diagrama
            raise ValueError("respuesta inválida")  # no reintentable: sin esperas
        return post_ok(path, payload, host, timeout)

    monkeypatch.setattr(C, "_ollama_post", post)
    r = C.run(tmp_path, workers=2)
    assert r["errores"] == 1 and r["ok"] == 7
    assert C.caption_state(a / "docA.md")["diagrama.png"] is False
    # El siguiente run solo reintenta lo que falló.
    monkeypatch.setattr(C, "_ollama_post", post_ok)
    r2 = C.run(tmp_path)
    assert r2["pendientes"] == 1 and r2["ok"] == 1


def test_run_falla_temprano_si_ollama_no_esta(tmp_path, monkeypatch):
    _doc_con_imagenes(tmp_path)
    monkeypatch.setattr(C, "CAPTION_MIN_PX", 64)

    def caido(*a, **k):
        raise C.OllamaError("no se pudo contactar Ollama")

    monkeypatch.setattr(C, "check_ollama", caido)
    with pytest.raises(C.OllamaError):
        C.run(tmp_path, backend="ollama")


def test_backend_disponible(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert C.backend_disponible("anthropic") == (False, "falta ANTHROPIC_API_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert C.backend_disponible("anthropic")[0] is True
    monkeypatch.setattr(C, "check_ollama", lambda *a, **k: None)
    assert C.backend_disponible("ollama")[0] is True

    def caido(*a, **k):
        raise C.OllamaError("primera línea\nsegunda")

    monkeypatch.setattr(C, "check_ollama", caido)
    assert C.backend_disponible("ollama") == (False, "primera línea")


def test_check_ollama_modelo(monkeypatch):
    monkeypatch.setattr(C, "ollama_models", lambda host=None: ["qwen3-vl:32b", "llava:latest"])
    C.check_ollama("qwen3-vl:32b")
    C.check_ollama("llava")
    with pytest.raises(C.OllamaError, match="no está"):
        C.check_ollama("otro:7b")


def test_dedup_captions_consecutivos(tmp_path):
    md = tmp_path / "d" / "doc.md"
    md.parent.mkdir()
    md.write_text(f"![](a.png)\n\n{MARCA} 1]*\n\n{MARCA} 2]*\n\nfin", encoding="utf-8")
    assert C.dedup(tmp_path) == {"duplicados_eliminados": 1, "archivos_modificados": 1}
    assert md.read_text(encoding="utf-8") == f"![](a.png)\n\n{MARCA} 1]*\n\nfin"
