r"""Parche para un bug conocido de surya-ocr 0.22.x (dependencia de marker-pdf).

Bug: datalab-to/surya#542 (https://github.com/datalab-to/surya/issues/542),
sin fix publicado en PyPI a la fecha. El schema JSON usado para forzar el
formato de bbox vía gramática GBNF de llama.cpp usa el escape regex `\d`, que
GBNF no soporta ("unknown escape"). Esto tumba el layout inference del modo
`--mode balanced` de Marker en TODAS las páginas (cae a un modo degradado sin
avisar claramente), y de paso lo vuelve mucho más lento por reintentos.

Fix: reemplazar `\d` por `[0-9]` en los dos patrones afectados de
surya/inference/prompts.py, siguiendo la solución sugerida en el issue.

Uso (con el venv activado, después de `pip install marker-pdf`):
    python patches/fix_surya_grammar.py

Es idempotente: si ya está parcheado, no hace nada. Hay que re-ejecutarlo
cada vez que se reinstale/actualice surya-ocr o marker-pdf.
"""
import sys
from pathlib import Path

try:
    import surya
except ImportError:
    print("surya-ocr no está instalado en este entorno (pip install marker-pdf).", file=sys.stderr)
    sys.exit(1)

def package_dir(module):
    """Directorio de un paquete, sea normal o de espacio de nombres.

    Un namespace package (sin __init__.py) tiene __file__ = None, así que
    `Path(module.__file__)` reventaría con TypeError; su ubicación real está
    en __path__.
    """
    if getattr(module, "__file__", None):
        return Path(module.__file__).parent
    for entry in getattr(module, "__path__", []):
        return Path(entry)
    raise RuntimeError(f"no se pudo localizar el paquete {module.__name__} en disco")


prompts_path = package_dir(surya) / "inference" / "prompts.py"
if not prompts_path.exists():
    print(f"No se encontró {prompts_path}; puede que la estructura del paquete haya cambiado.", file=sys.stderr)
    sys.exit(1)

text = prompts_path.read_text(encoding="utf-8")
old = r'r"^\d{1,4} \d{1,4} \d{1,4} \d{1,4}$"'
new = r'r"^[0-9]{1,4} [0-9]{1,4} [0-9]{1,4} [0-9]{1,4}$"'

n = text.count(old)
if n == 0:
    if new in text:
        print("Ya está parcheado, nada que hacer.")
        sys.exit(0)
    # Salida NO cero a propósito: si el patrón desapareció (versión distinta a
    # la pineada en requirements.txt), el parche no se aplicó y Marker correría
    # degradado sin avisar. Mejor romper el setup/build aquí que descubrirlo
    # tras un lote de 40 horas.
    print(f"ERROR: no se encontró el patrón esperado en {prompts_path}.\n"
          f"       surya-ocr instalado: {getattr(surya, '__version__', 'desconocido')} "
          f"(requirements.txt pinea 0.22.1).\n"
          f"       Revisar el estado del bug upstream antes de continuar: "
          f"https://github.com/datalab-to/surya/issues/542", file=sys.stderr)
    sys.exit(1)

text = text.replace(old, new)
prompts_path.write_text(text, encoding="utf-8")
print(f"Parche aplicado en {prompts_path} ({n} ocurrencias reemplazadas).")
