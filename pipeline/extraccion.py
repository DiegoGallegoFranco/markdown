"""Extracción DETALLADA del contenido de las imágenes (CAPTION_MODE=detallado).

Por qué: una descripción de 1-2 frases hace una imagen *encontrable* ("tabla de
presupuesto"), pero pierde sus DATOS. Si una cifra solo existe dentro de una
tabla pegada como imagen, el banco de conocimiento no puede responder sobre
ella. Aquí el modelo identifica el tipo de imagen y extrae lo que corresponde a
ese tipo (la tabla completa, la transcripción literal, las relaciones de un
diagrama...) en UNA sola llamada, con salida JSON validada por esquema.

Este módulo no hace E/S con el modelo ni escribe en los .md: construye el
prompt, interpreta la respuesta, calcula el contexto y la página, renderiza el
bloque Markdown y gestiona la caché. captions.py orquesta.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from .mdimages import es_descripcion_ia, image_refs, is_external, ref_name

# Cambiarlo invalida la caché: respuestas obtenidas con otro prompt no se reusan.
PROMPT_VERSION = "2026-11-24.1"

TIPOS = ["tabla", "grafica", "diagrama", "texto", "formulario", "mapa",
         "foto", "captura", "logo", "decorativa", "otro"]
TIPO_ETIQUETA = {
    "tabla": "Tabla", "grafica": "Gráfica", "diagrama": "Diagrama",
    "texto": "Texto", "formulario": "Formulario", "mapa": "Mapa",
    "foto": "Foto", "captura": "Captura de pantalla", "logo": "Logo",
    "decorativa": "Decorativa", "otro": "Imagen",
}
# Tipos sin información que extraer: bloque mínimo (ver render_bloque).
TIPOS_SIN_CONTENIDO = {"logo", "decorativa"}

# JSON Schema que Ollama impone en la salida (`format`). Con él la respuesta
# llega siempre completa y con las mismas claves, en vez de texto libre que
# habría que adivinar cómo partir.
ESQUEMA = {
    "type": "object",
    "properties": {
        "tipo": {"type": "string", "enum": TIPOS},
        "titulo": {"type": "string"},
        "resumen": {"type": "string"},
        "contenido": {"type": "string"},
        "texto_visible": {"type": "string"},
        "palabras_clave": {"type": "array", "items": {"type": "string"}},
        "entidades": {"type": "array", "items": {"type": "string"}},
        "calidad": {"type": "string", "enum": ["alta", "media", "baja"]},
    },
    "required": ["tipo", "titulo", "resumen", "contenido", "texto_visible",
                 "palabras_clave", "entidades", "calidad"],
}

PROMPT = """Eres un sistema de extracción de información de imágenes de documentos (informes, manuales, actas, presentaciones). Tu salida se indexará en un buscador: el objetivo es que TODA la información de la imagen quede como texto buscable, porque puede que no exista en ningún otro lugar del documento.

Primero identifica el tipo de imagen y después extrae según el tipo:
- tabla: reconstruye la tabla COMPLETA en Markdown en "contenido", con todas las filas, columnas, cifras y unidades exactamente como aparecen. No resumas ni omitas filas.
- grafica: en "contenido" indica tipo de gráfica, título, ejes con sus unidades, cada serie con sus valores (exactos si están rotulados; si los estimas, márcalos con "≈") y la tendencia principal.
- diagrama (organigrama, flujo, esquema, arquitectura): en "contenido" lista los elementos y sus relaciones, una por línea, con la forma "A → B (etiqueta)". Respeta la jerarquía.
- texto, formulario o captura: transcribe LITERALMENTE todo el texto en "contenido", conservando el orden y la estructura (campos y valores de un formulario como "Campo: valor").
- mapa o plano: en "contenido" lista lugares, rótulos, leyenda, escala y orientación visibles.
- foto: describe lo relevante en "resumen" y pon en "texto_visible" cualquier texto (rótulos, placas, letreros).
- logo o decorativa (sin información útil): solo "tipo" y un "titulo" breve; deja el resto vacío.

Reglas:
- No inventes. Si algo no se lee, escribe [ilegible]. Copia cifras, códigos, fechas, siglas y nombres propios exactamente.
- Escribe "titulo" y "resumen" en español. Transcribe el texto de la imagen en su idioma original.
- "resumen": 1-3 frases que expliquen qué muestra la imagen y para qué sirve.
- "texto_visible": texto de la imagen que no esté ya en "contenido" (vacío si no hay).
- "palabras_clave": 5-12 términos que alguien usaría para buscar esta imagen.
- "entidades": personas, cargos, unidades u organizaciones, lugares, fechas, códigos y cifras clave.
- "calidad": "baja" si la imagen es borrosa o tu lectura es dudosa.
Responde únicamente con el objeto JSON."""


# --------------------------------------------------------------------------- contexto

_RE_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_RE_PIE = re.compile(r"^\s*[*_]*\s*(figura|fig\.|tabla|gr[aá]fic[oa]|imagen|mapa|diagrama|"
                     r"esquema|cuadro|ilustraci[oó]n|fuente)\b", re.IGNORECASE)


def _limpio(texto: str, max_chars: int) -> str:
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto if len(texto) <= max_chars else texto[: max_chars - 1] + "…"


def _es_imagen(parrafo: str) -> bool:
    return bool(image_refs(parrafo))


def orden_imagenes(md_text: str) -> list[str]:
    """Nombres de imagen locales en orden de PRIMERA aparición (para numerar)."""
    vistos: list[str] = []
    for ref in image_refs_ordenadas(md_text):
        if is_external(ref):
            continue
        nombre = ref_name(ref)
        if nombre not in vistos:
            vistos.append(nombre)
    return vistos


def image_refs_ordenadas(md_text: str) -> list[str]:
    # image_refs() devuelve primero todas las ![]() y luego las <img>; para
    # numerar hace falta el orden real del documento, así que se recorre por
    # párrafos (dentro de un párrafo el orden mixto es irrelevante en la práctica).
    refs: list[str] = []
    for parrafo in md_text.split("\n\n"):
        refs.extend(image_refs(parrafo))
    return refs


def contexto(md_text: str, img_name: str) -> dict:
    """Sección, texto anterior/posterior y número de la imagen en el documento.

    El contexto se envía al modelo (le dice, por ejemplo, que una tabla sin
    título es "Presupuesto 2025" porque lo dice el párrafo de arriba) y la
    sección y el número quedan en el bloque y en el índice.
    """
    parrafos = md_text.split("\n\n")
    pos = None
    for i, p in enumerate(parrafos):
        if any(ref_name(r) == img_name for r in image_refs(p)):
            pos = i
            break
    orden = orden_imagenes(md_text)
    numero = orden.index(img_name) + 1 if img_name in orden else len(orden) + 1
    if pos is None:
        return {"numero": numero, "seccion": "", "antes": "", "despues": ""}

    pila: list[tuple[int, str]] = []
    for p in parrafos[:pos]:
        for linea in p.splitlines():
            if m := _RE_HEADING.match(linea.strip()):
                nivel, titulo = len(m.group(1)), _limpio(m.group(2), 120)
                while pila and pila[-1][0] >= nivel:
                    pila.pop()
                pila.append((nivel, titulo))
    seccion = " > ".join(t for _, t in pila)

    def util(p: str) -> bool:
        return p.strip() and not _es_imagen(p) and not es_descripcion_ia(p) \
            and not _RE_HEADING.match(p.strip())

    antes = next((p for p in reversed(parrafos[:pos]) if util(p)), "")
    # Si el párrafo siguiente es un pie de figura ("Figura 3. ..."), es el
    # mejor título posible; si no, se toma el siguiente párrafo de texto.
    siguientes = [p for p in parrafos[pos + 1: pos + 4] if util(p)]
    despues = siguientes[0] if siguientes else ""
    return {
        "numero": numero,
        "seccion": seccion,
        "antes": _limpio(antes, 600),
        "despues": _limpio(despues, 400),
        "pie": _limpio(despues, 300) if despues and _RE_PIE.match(despues) else "",
    }


def prompt_con_contexto(ctx: dict) -> str:
    partes = []
    if ctx.get("seccion"):
        partes.append(f"Sección del documento: {ctx['seccion']}")
    if ctx.get("pie"):
        partes.append(f"Pie de figura: {ctx['pie']}")
    if ctx.get("antes"):
        partes.append(f"Texto anterior a la imagen: {ctx['antes']}")
    if ctx.get("despues") and not ctx.get("pie"):
        partes.append(f"Texto posterior a la imagen: {ctx['despues']}")
    if not partes:
        return PROMPT
    return (PROMPT + "\n\nContexto del documento (úsalo para interpretar la imagen y "
            "darle título; NO lo copies como contenido de la imagen):\n" + "\n".join(partes))


# --------------------------------------------------------------------------- página

_RE_PAG_PYMUPDF = re.compile(r"-(\d{4})-\d+\.\w+$")   # doc.pdf-0002-01.png (1-based)
_RE_PAG_MARKER = re.compile(r"_page_(\d+)_")          # _page_1_Picture_3.jpeg (0-based)
_RE_PAG_PPTX = re.compile(r"^slide(\d{3})-\d{3}\.")   # slide004-002.png (1-based)


def pagina_de(img_name: str) -> tuple[int | None, str]:
    """(número, unidad) deducidos del nombre que pone cada conversor.

    Comprobado empíricamente: pymupdf4llm 1.28 numera desde 1 con 4 dígitos;
    marker 2.0.0 usa el page_id interno (desde 0); el conversor pptx propio
    numera diapositivas desde 1. pandoc (docx) no codifica la página: None.
    """
    if m := _RE_PAG_PPTX.match(img_name):
        return int(m.group(1)), "diapositiva"
    if m := _RE_PAG_MARKER.search(img_name):
        return int(m.group(1)) + 1, "página"
    if m := _RE_PAG_PYMUPDF.search(img_name):
        return int(m.group(1)), "página"
    return None, "página"


# --------------------------------------------------------------------------- respuesta del modelo

def _lista(v) -> list[str]:
    if isinstance(v, str):
        v = re.split(r"[,;\n]", v)
    if not isinstance(v, list):
        return []
    out, vistos = [], set()
    for x in v:
        s = _limpio(str(x), 80)
        if s and s.lower() not in vistos:
            vistos.add(s.lower())
            out.append(s)
    return out[:15]


def normalizar(datos: dict) -> dict:
    tipo = str(datos.get("tipo", "")).strip().lower()
    tipo = {"gráfica": "grafica", "grafico": "grafica", "gráfico": "grafica"}.get(tipo, tipo)
    calidad = str(datos.get("calidad", "")).strip().lower()
    return {
        "tipo": tipo if tipo in TIPOS else "otro",
        "titulo": _limpio(str(datos.get("titulo") or ""), 160),
        "resumen": _limpio(str(datos.get("resumen") or ""), 800),
        "contenido": str(datos.get("contenido") or "").strip(),
        "texto_visible": str(datos.get("texto_visible") or "").strip(),
        "palabras_clave": _lista(datos.get("palabras_clave")),
        "entidades": _lista(datos.get("entidades")),
        "calidad": calidad if calidad in ("alta", "media", "baja") else "media",
    }


def interpretar(texto: str) -> tuple[dict, bool]:
    """(datos normalizados, ok). Si el JSON no es válido —p. ej. la respuesta se
    cortó en una tabla enorme— no se pierde lo leído: se guarda como contenido
    con calidad baja (ok=False, para no cachearlo ni darlo por bueno)."""
    texto = (texto or "").strip()
    candidatos = [texto]
    if "{" in texto and "}" in texto:
        candidatos.append(texto[texto.index("{"): texto.rindex("}") + 1])
    for c in candidatos:
        try:
            datos = json.loads(c)
        except (ValueError, TypeError):
            continue
        if isinstance(datos, dict):
            return normalizar(datos), True
    return normalizar({"tipo": "otro", "titulo": "", "contenido": texto,
                       "resumen": "Extracción incompleta: la respuesta del modelo no "
                                  "tenía el formato esperado.", "calidad": "baja"}), False


# --------------------------------------------------------------------------- bloque Markdown

def _seguro(texto: str) -> str:
    # Dentro del bloque no puede haber nada que parezca una referencia a imagen:
    # caption_state() lo tomaría por otra imagen del documento.
    return texto.replace("![", "!\\[").replace("<img", "&lt;img")


def _una_linea(texto: str) -> str:
    return _seguro(re.sub(r"\s*\n\s*", " / ", texto.strip()))


def render_bloque(datos: dict, numero: int, ident: str, pagina: int | None,
                  unidad: str, seccion: str) -> str:
    """Bloque de cita de UN solo párrafo (ver mdimages.BLOQUE_*): todas las
    líneas empiezan por `>` y las vacías son `>`, nunca una línea en blanco."""
    etiqueta = TIPO_ETIQUETA.get(datos["tipo"], "Imagen")
    titulo = _una_linea(datos["titulo"]) or "(sin título)"
    meta = [f"id: {ident}"]
    if pagina:
        meta.append(f"{unidad} {pagina}")
    if seccion:
        meta.append(f"sección: {_una_linea(seccion)}")
    lineas = [f"> **Imagen {numero} · {etiqueta}** — {titulo}",
              f"> *Contenido extraído por IA · {' · '.join(meta)}*"]

    if datos["tipo"] in TIPOS_SIN_CONTENIDO:
        return "\n".join(lineas)

    def parrafo(texto: str):
        lineas.append(">")
        for linea in _seguro(texto).splitlines():
            lineas.append(f"> {linea}".rstrip() if linea.strip() else ">")

    if datos["resumen"]:
        parrafo(datos["resumen"])
    if datos["contenido"]:
        parrafo(datos["contenido"])
    extras = []
    if datos["texto_visible"]:
        extras.append(f"> **Texto visible:** {_una_linea(datos['texto_visible'])}")
    if datos["palabras_clave"]:
        extras.append(f"> **Palabras clave:** {_una_linea(', '.join(datos['palabras_clave']))}")
    if datos["entidades"]:
        extras.append(f"> **Entidades:** {_una_linea(', '.join(datos['entidades']))}")
    if datos["calidad"] == "baja":
        extras.append("> **Aviso:** lectura dudosa (imagen poco legible); verificar contra el original.")
    if extras:
        lineas.append(">")
        lineas.extend(extras)
    return "\n".join(lineas)


# --------------------------------------------------------------------------- caché

def dir_cache_por_defecto() -> Path:
    # En Docker DATA_DIR=/data vive en el volumen: la caché sobrevive a las
    # recreaciones del contenedor y se comparte entre trabajos de la web.
    if os.environ.get("DATA_DIR"):
        return Path(os.environ["DATA_DIR"]) / "caption-cache"
    return Path.home() / ".cache" / "doc2md" / "captions"


def clave_cache(sha256: str, modelo: str, max_px: int) -> str:
    firma = hashlib.sha256(f"{modelo}|{max_px}|{PROMPT_VERSION}".encode()).hexdigest()[:12]
    return f"{sha256}-{firma}"


def cache_get(directorio: Path | None, clave: str) -> dict | None:
    if not directorio:
        return None
    try:
        return json.loads((Path(directorio) / f"{clave}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def cache_put(directorio: Path | None, clave: str, datos: dict) -> None:
    """Escritura atómica (tmp + replace): dos hilos o dos trabajos escribiendo
    la misma clave no dejan nunca un JSON a medias."""
    if not directorio:
        return
    try:
        d = Path(directorio)
        d.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False)
        os.replace(tmp, d / f"{clave}.json")
    except OSError:
        pass  # la caché es una optimización: sin ella solo se repite trabajo
