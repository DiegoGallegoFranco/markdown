"""Núcleo del pipeline PDF/DOCX/PPTX -> Markdown.

Los scripts de la raíz del repo y la app web (webapp/) son ambos clientes
delgados de este paquete: toda la lógica de conversión vive aquí para que la
CLI y el servidor no se desincronicen.
"""

__all__ = ["paths", "inventory", "converters", "captions", "qc"]
