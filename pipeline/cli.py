"""CLI unificada del pipeline.

    python -m pipeline inventario  /ruta/a/documentos
    python -m pipeline convertir   /ruta/a/documentos --engine auto
    python -m pipeline captions    --dry-run
    python -m pipeline qc
    python -m pipeline todo        /ruta/a/documentos --engine auto --captions

Los scripts sueltos de la raíz del repo (convertir_pymupdf.py, etc.) siguen
funcionando: son wrappers de estos subcomandos.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import captions as captions_mod
from . import converters, inventory, qc
from .paths import PDF_EXTS, collect_documents, common_root


def _log(msg):
    print(msg, flush=True)


def _expand(targets) -> list[Path]:
    docs: list[Path] = []
    for t in targets:
        found = collect_documents(t)
        if not found:
            print(f"AVISO: sin documentos soportados en {t}", file=sys.stderr)
        docs.extend(found)
    # dedup preservando orden
    return list(dict.fromkeys(docs))


def _engine_map(docs, engine, log=_log) -> dict[Path, str]:
    """Motor de PDF por documento. `auto` decide con el inventario."""
    pdfs = [d for d in docs if d.suffix.lower() in PDF_EXTS]
    if engine != "auto" or not pdfs:
        return {d: engine if engine != "auto" else "pymupdf" for d in pdfs}

    log(f"Modo auto: analizando {len(pdfs)} PDF(s) para elegir motor...")
    rows = inventory.build_inventory(pdfs, common_root(pdfs))
    mapping = {}
    for pdf, row in zip(pdfs, rows):
        mapping[pdf] = row.get("motor_sugerido") or "pymupdf"
    escaneados = sum(1 for v in mapping.values() if v == "marker")
    log(f"Modo auto: {escaneados} a Marker (escaneados/OCR), {len(pdfs)-escaneados} a pymupdf4llm")
    return mapping


def cmd_inventario(args) -> int:
    root = Path(args.ruta)
    pdfs = collect_documents(root, exts=PDF_EXTS)
    if not pdfs:
        print(f"No se encontraron PDFs en {root}", file=sys.stderr)
        return 1
    rows = inventory.build_inventory(pdfs, common_root(pdfs), on_progress=_log)
    out = Path(args.csv) if args.csv else Path.cwd() / "inventario.csv"
    inventory.write_csv(rows, out)
    print(f"\nInventario generado: {out} ({len(rows)} PDFs)")
    return 0


def cmd_convertir(args) -> int:
    docs = _expand(args.rutas)
    if not docs:
        print("Nada que convertir.", file=sys.stderr)
        return 1
    input_root = Path(args.input_root) if args.input_root else common_root(docs)
    engines = _engine_map(docs, args.engine)

    results = []
    for n, src in enumerate(docs, start=1):
        _log(f"[{n}/{len(docs)}] {src.name}")
        pdf_engine = engines.get(src, "pymupdf")
        results.append(converters.convert(
            src, args.out, input_root, pdf_engine=pdf_engine,
            skip_existing=args.skip_existing, log=_log,
        ))

    ok = sum(1 for r in results if r["ok"])
    print(f"\nLote completo: {ok} OK, {len(results)-ok} con error.")
    for r in results:
        if not r["ok"]:
            print(f"  ! {Path(r['src']).name}: {r['error']}", file=sys.stderr)
    return 0 if ok else 1


def cmd_captions(args) -> int:
    imgs = None
    if args.from_file:
        imgs = [Path(l.strip()) for l in Path(args.from_file).read_text().splitlines() if l.strip()]
    try:
        captions_mod.run(args.output, model=args.model, workers=args.workers,
                         limit=args.limit, images=imgs, log=_log, dry_run=args.dry_run)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_captions_audit(args) -> int:
    faltantes = captions_mod.audit(args.output)
    print(f"Imágenes sin caption: {len(faltantes)}")
    for f in faltantes:
        print(f"  {f}")
    if faltantes:
        dest = Path.cwd() / "missing_captions.txt"
        dest.write_text("\n".join(faltantes), encoding="utf-8")
        print(f"\nGuardado en {dest} — reprocesar con:\n"
              f"  python -m pipeline captions --from-file {dest}")
    return 0


def cmd_captions_dedup(args) -> int:
    r = captions_mod.dedup(args.output, log=_log)
    print(f"\nTotal: {r['duplicados_eliminados']} captions duplicados eliminados "
          f"en {r['archivos_modificados']} archivos")
    return 0


def cmd_qc(args) -> int:
    output_dir = Path(args.output)
    if not output_dir.exists():
        print(f"No existe {output_dir}", file=sys.stderr)
        return 1
    rows = qc.run(output_dir)
    out = Path(args.csv) if args.csv else Path.cwd() / "inventario_final.csv"
    qc.write_csv(rows, out)
    s = qc.summarize(rows)
    print(f"{out} generado ({s['documentos']} documentos)")
    print(f"  OK: {s['ok']} | Revisión manual: {s['revision']} | Error: {s['error']}")
    print(f"  Imágenes: {s['imagenes']} | Captions: {s['captions']}")
    if s["revision"] or s["error"]:
        print("\nDocumentos que requieren atención:")
        for r in rows:
            if r["estado"] != "OK":
                print(f"  ! {r['ruta_md']}: {r['estado']}")
    return 0


def cmd_todo(args) -> int:
    """Pipeline completo: inventario -> conversión -> captions -> QC."""
    rc = cmd_convertir(args)
    if args.captions:
        print("\n=== Captioning de imágenes ===")
        cmd_captions(argparse.Namespace(
            output=args.out, model=args.model, workers=args.workers,
            limit=None, from_file=None, dry_run=False,
        ))
    print("\n=== Control de calidad ===")
    cmd_qc(argparse.Namespace(output=args.out, csv=None))
    return rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_convert_args(sp):
        sp.add_argument("rutas", nargs="+", help="Archivos o directorios de entrada")
        sp.add_argument("--out", default="output", help="Directorio de salida (default: output)")
        sp.add_argument("--input-root", default=None,
                        help="Raíz para calcular rutas relativas (default: ancestro común del lote)")
        sp.add_argument("--engine", choices=["auto", "pymupdf", "marker"], default="auto",
                        help="Motor para PDFs; auto elige por documento según el inventario")
        sp.add_argument("--skip-existing", action="store_true",
                        help="No reconvertir documentos que ya tienen un .md no vacío")

    sp = sub.add_parser("inventario", help="Fase 1: clasificar PDFs (nativo/escaneado)")
    sp.add_argument("ruta")
    sp.add_argument("--csv", default=None)
    sp.set_defaults(func=cmd_inventario)

    sp = sub.add_parser("convertir", help="Fases 2-3: convertir a Markdown")
    add_convert_args(sp)
    sp.set_defaults(func=cmd_convertir)

    sp = sub.add_parser("captions", help="Describir imágenes con la API de Claude")
    sp.add_argument("--output", default="output")
    sp.add_argument("--model", default=captions_mod.DEFAULT_MODEL)
    sp.add_argument("--workers", type=int, default=captions_mod.DEFAULT_WORKERS)
    sp.add_argument("--limit", type=int, default=None, help="Smoke test: procesar solo N")
    sp.add_argument("--from-file", default=None, help="Archivo con una ruta de imagen por línea")
    sp.add_argument("--dry-run", action="store_true", help="Solo contar, no llamar a la API")
    sp.set_defaults(func=cmd_captions)

    sp = sub.add_parser("captions-audit", help="Listar imágenes sin caption")
    sp.add_argument("--output", default="output")
    sp.set_defaults(func=cmd_captions_audit)

    sp = sub.add_parser("captions-dedup", help="Eliminar captions duplicados consecutivos")
    sp.add_argument("--output", default="output")
    sp.set_defaults(func=cmd_captions_dedup)

    sp = sub.add_parser("qc", help="Fase 4: control de calidad e inventario final")
    sp.add_argument("--output", default="output")
    sp.add_argument("--csv", default=None)
    sp.set_defaults(func=cmd_qc)

    sp = sub.add_parser("todo", help="Pipeline completo de principio a fin")
    add_convert_args(sp)
    sp.add_argument("--captions", action="store_true", help="Incluir el paso de captioning")
    sp.add_argument("--model", default=captions_mod.DEFAULT_MODEL)
    sp.add_argument("--workers", type=int, default=captions_mod.DEFAULT_WORKERS)
    sp.set_defaults(func=cmd_todo)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
