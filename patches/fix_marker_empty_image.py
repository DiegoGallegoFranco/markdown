"""Parche defensivo para un bug de marker-pdf: crashea al guardar imágenes
de tamaño cero.

Bug: algunas cajas de layout se recortan a una región de área cero (bbox
degenerado del modelo de layout). marker/output.py intenta guardar esa
imagen vacía con Pillow, que revienta con:

    ValueError: cannot write empty image as JPEG

Esto aborta la conversión COMPLETA del documento (aunque ya llevara 20+
minutos procesado), sin ningún workaround por CLI. No hay issue/fix
upstream conocido a la fecha.

Fix: en save_output(), saltar imágenes con width==0 o height==0 (con un
aviso), en vez de abortar todo el documento.

Uso (con el venv activado, después de `pip install marker-pdf`):
    python patches/fix_marker_empty_image.py

Es idempotente. Hay que re-ejecutarlo cada vez que se reinstale/actualice
marker-pdf.
"""
import sys
from pathlib import Path

try:
    import marker
except ImportError:
    print("marker-pdf no está instalado en este entorno (pip install marker-pdf).", file=sys.stderr)
    sys.exit(1)

def package_dir(module):
    """Directorio de un paquete, sea normal o de espacio de nombres.

    `marker` es un namespace package (no trae __init__.py), así que su
    __file__ es None y `Path(module.__file__)` revienta con TypeError. La
    ubicación real está en __path__.
    """
    if getattr(module, "__file__", None):
        return Path(module.__file__).parent
    for entry in getattr(module, "__path__", []):
        return Path(entry)
    raise RuntimeError(f"no se pudo localizar el paquete {module.__name__} en disco")


output_path = package_dir(marker) / "output.py"
if not output_path.exists():
    print(f"No se encontró {output_path}; puede que la estructura del paquete haya cambiado.", file=sys.stderr)
    sys.exit(1)

text = output_path.read_text(encoding="utf-8")
old = (
    "    for img_name, img in images.items():\n"
    "        img = convert_if_not_rgb(img)  # RGBA images can't save as JPG\n"
    "        img.save(os.path.join(output_dir, img_name), settings.OUTPUT_IMAGE_FORMAT)"
)
new = (
    "    for img_name, img in images.items():\n"
    "        if img.width == 0 or img.height == 0:\n"
    "            # Algunas cajas de layout se recortan a área cero; Pillow no\n"
    "            # puede guardar eso como JPEG. Saltar en vez de abortar todo\n"
    "            # el documento.\n"
    "            print(f\"Warning: skipping empty image '{img_name}' (size {img.size})\")\n"
    "            continue\n"
    "        img = convert_if_not_rgb(img)  # RGBA images can't save as JPG\n"
    "        img.save(os.path.join(output_dir, img_name), settings.OUTPUT_IMAGE_FORMAT)"
)

if new in text:
    print("Ya está parcheado, nada que hacer.")
    sys.exit(0)

n = text.count(old)
if n == 0:
    # Salida NO cero a propósito: sin este parche, un bbox degenerado aborta la
    # conversión COMPLETA del documento. Que el setup/build falle aquí es mucho
    # más barato que perder 20+ minutos de proceso por documento.
    print(f"ERROR: no se encontró el patrón esperado en {output_path}.\n"
          f"       marker-pdf instalado: {getattr(marker, '__version__', 'desconocido')} "
          f"(requirements.txt pinea 2.0.0).\n"
          f"       Revisar si el bug de 'cannot write empty image as JPEG' sigue vigente "
          f"antes de continuar.", file=sys.stderr)
    sys.exit(1)

text = text.replace(old, new)
output_path.write_text(text, encoding="utf-8")
print(f"Parche aplicado en {output_path}.")
