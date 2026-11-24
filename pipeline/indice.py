"""Índice de imágenes: por documento (imagenes.json) y global (raíz de la salida).

La fuente de verdad es el `imagenes.json` de cada documento, que captions.py
escribe al insertar cada bloque detallado. A partir de ellos se generan, en la
raíz de la salida:

    INDICE_IMAGENES.md     legible: agrupado por tipo y documento, con enlaces
    indice_imagenes.csv    para filtrar/ordenar en una hoja de cálculo
    indice_imagenes.json   todo junto, para que otro sistema (p. ej. el
                           ingestor del banco de conocimiento) lo cargue

Regenerarlo es barato e idempotente: solo lee JSON.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

from .extraccion import TIPO_ETIQUETA, TIPOS, TIPOS_SIN_CONTENIDO

SIDECAR = "imagenes.json"
CAMPOS_CSV = ["id", "tipo", "titulo", "documento", "pagina", "unidad_pagina",
              "seccion", "resumen", "palabras_clave", "entidades", "calidad",
              "ruta", "reutilizada_de"]


def escribir_json_atomico(ruta: Path, datos) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=ruta.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)
    os.replace(tmp, ruta)


def leer_sidecar(ruta: Path) -> list[dict]:
    try:
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        return datos if isinstance(datos, list) else []
    except (OSError, ValueError):
        return []


def upsert_sidecar(doc_dir: Path, entrada: dict) -> None:
    """Inserta o reemplaza la entrada de una imagen (clave: `archivo`).

    Lo llama captions.py DENTRO del lock del .md del documento, así que dos
    hilos no se pisan el mismo imagenes.json.
    """
    ruta = Path(doc_dir) / SIDECAR
    entradas = [e for e in leer_sidecar(ruta) if e.get("archivo") != entrada["archivo"]]
    entradas.append(entrada)
    entradas.sort(key=lambda e: (e.get("md", ""), e.get("numero", 0)))
    escribir_json_atomico(ruta, entradas)


def _celda(texto) -> str:
    return str(texto or "").replace("|", "\\|").replace("\n", " ").strip()


def construir(output_dir) -> dict:
    """Regenera el índice global. Devuelve {imagenes, documentos, rutas}."""
    output_dir = Path(output_dir)
    filas = []
    for sidecar in sorted(output_dir.rglob(SIDECAR)):
        doc_dir = sidecar.parent
        for e in leer_sidecar(sidecar):
            ruta = (doc_dir / e.get("archivo", "")).relative_to(output_dir).as_posix()
            filas.append({**e, "ruta": ruta})

    orden_tipo = {t: i for i, t in enumerate(TIPOS)}
    filas.sort(key=lambda f: (orden_tipo.get(f.get("tipo"), 99), f.get("documento", ""),
                              f.get("numero", 0)))

    # JSON completo (incluye `contenido`: tablas y transcripciones enteras).
    escribir_json_atomico(output_dir / "indice_imagenes.json", filas)

    with open(output_dir / "indice_imagenes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_CSV, extrasaction="ignore")
        w.writeheader()
        for fila in filas:
            w.writerow({**fila,
                        "palabras_clave": "; ".join(fila.get("palabras_clave") or []),
                        "entidades": "; ".join(fila.get("entidades") or [])})

    documentos = sorted({f.get("documento", "") for f in filas})
    lineas = ["# Índice de imágenes", "",
              f"{len(filas)} imágenes en {len(documentos)} documento(s). "
              "Contenido extraído por IA: verificar cifras críticas contra el original.", ""]
    resumen_tipos = {}
    for f in filas:
        resumen_tipos[f.get("tipo")] = resumen_tipos.get(f.get("tipo"), 0) + 1
    lineas += ["| Tipo | Imágenes |", "|---|---|"]
    lineas += [f"| {TIPO_ETIQUETA.get(t, t)} | {n} |" for t, n in
               sorted(resumen_tipos.items(), key=lambda kv: orden_tipo.get(kv[0], 99))]

    # Logos y decorativas solo cuentan en el resumen: listarlos (uno por página
    # en muchos documentos) taparía las imágenes con información. Siguen
    # completos en el CSV y el JSON.
    sin_info = [f for f in filas if f.get("tipo") in TIPOS_SIN_CONTENIDO]
    if sin_info:
        lineas += ["", f"*{len(sin_info)} logo(s)/imagen(es) decorativa(s) no se listan "
                       "aquí; están en indice_imagenes.csv.*"]

    tipo_actual = None
    for f in filas:
        if f.get("tipo") in TIPOS_SIN_CONTENIDO:
            continue
        if f.get("tipo") != tipo_actual:
            tipo_actual = f.get("tipo")
            lineas += ["", f"## {TIPO_ETIQUETA.get(tipo_actual, tipo_actual)}", "",
                       "| Id | Título | Documento | Pág. | Sección | Palabras clave |",
                       "|---|---|---|---|---|---|"]
        pag = f.get("pagina") or ""
        if pag and f.get("unidad_pagina") == "diapositiva":
            pag = f"diap. {pag}"
        lineas.append(
            f"| [{_celda(f.get('id'))}](<{f['ruta']}>) | {_celda(f.get('titulo')) or '—'} "
            f"| {_celda(f.get('documento'))} | {pag} | {_celda(f.get('seccion'))} "
            f"| {_celda(', '.join(f.get('palabras_clave') or []))} |")
    (output_dir / "INDICE_IMAGENES.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")

    return {"imagenes": len(filas), "documentos": len(documentos),
            "rutas": [str(output_dir / n) for n in
                      ("INDICE_IMAGENES.md", "indice_imagenes.csv", "indice_imagenes.json")]}
