# pdf-a-md

Pipeline para convertir PDFs (y docx/pptx) a Markdown limpio, con imágenes
extraídas y enlazadas con rutas relativas. Se usa de dos formas:

- **CLI** — `python -m pipeline ...` sobre una carpeta de documentos propia
  (que **no** vive en este repo; ver `.gitignore`).
- **Web** — subir documentos por el navegador y descargar el Markdown en un ZIP.
  Pensada para desplegarse en un servidor (ver [Docker](#despliegue-con-docker)).

Dos motores para PDF, con trade-offs distintos:

- **`pymupdf4llm`**: rápido, sin GPU, sin dependencias pesadas. Buen resultado
  en PDFs de una columna con tablas simples. Falla en layouts multicolumna
  complejos: mezcla el texto de columnas paralelas en un solo párrafo. **No hace
  OCR**: sobre un PDF escaneado devuelve un documento vacío.
- **`Marker`** (Datalab): el único de los dos que hace **OCR real**. Resuelve
  multicolumna y genera mejor jerarquía de encabezados en documentos densos
  (manuales, doctrina, informes largos). Mucho más lento y con más setup.

  Marker tiene dos modos y **elige solo según el dispositivo**:

  | Modo | Qué hace | Dónde corre |
  |---|---|---|
  | `fast` | Detectores CPU ligeros para layout/tablas; OCR solo de bloques ilegibles | Default en **CPU/MPS** |
  | `balanced` | Modelo de layout VLM + OCR de página completa; mejor calidad | Default en **GPU**. Requiere el binario `llama-server` |

  Esto importa para la decisión de hardware: en un servidor **sin GPU no solo
  vas más lento, corres un modo distinto y de menor calidad**.

El **modo `auto`** (por defecto) hace la elección por documento: clasifica cada
PDF con el inventario y manda a Marker solo los escaneados. Es lo que conviene
en casi todos los casos.

## Arquitectura

```
pipeline/          núcleo: rutas, inventario, conversores, captions, QC
  paths.py         resolución de rutas de entrada/salida
  inventory.py     fase 1 — clasificar nativo vs escaneado
  converters.py    pymupdf4llm / Marker / pandoc / python-pptx
  captions.py      descripción de imágenes con la API de Claude (reanudable)
  qc.py            fase 4 — validación e inventario final
  cli.py           CLI unificada (python -m pipeline)
webapp/            interfaz web (FastAPI) + cola de trabajos (SQLite)
patches/           parches a bugs upstream de marker-pdf y surya-ocr
*.py, *.sh         wrappers de compatibilidad de los comandos originales
```

La CLI y la web comparten exactamente el mismo código de conversión; no hay dos
implementaciones que puedan desincronizarse.

## Setup local

Requiere **Python 3.10+** (marker-pdf y FastAPI no soportan 3.9; el `python3`
del sistema en macOS suele ser 3.9 — usar `brew install python@3.12`).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # núcleo (incluye Marker)
pip install -r requirements-web.txt    # solo si vas a usar la interfaz web

# Alternativa sin Marker (sin torch, sin OCR — mucho más liviana):
# pip install -r requirements-light.txt

# Necesario para el modo `balanced` de Marker (el de mayor calidad), en
# cualquier plataforma — no solo en Mac:
brew install llama.cpp

# Solo si vas a convertir .docx:
brew install pandoc

# Aplicar SIEMPRE después de instalar/actualizar marker-pdf, antes de usar
# --mode balanced (ver detalle de los bugs en patches/*.py). Ahora SALEN CON
# ERROR si el patrón esperado no está — no fallan en silencio:
python patches/fix_surya_grammar.py
python patches/fix_marker_empty_image.py
```

> **Las versiones de `requirements.txt` están pineadas a propósito.** Los parches
> reescriben líneas literales de `marker-pdf 2.0.0` y `surya-ocr 0.22.1`. Al
> subir cualquiera de las dos, revisar `patches/` **antes** de desplegar.

## Uso por CLI

```bash
# Fase 1 — inventario y diagnóstico
python -m pipeline inventario /ruta/a/documentos

# Fase 2 — piloto sobre unos pocos documentos representativos
python -m pipeline convertir "/ruta/doc1.pdf" "/ruta/doc2.pdf" --engine auto

# Fase 3 — lote completo (acepta archivos o directorios, rutas absolutas)
python -m pipeline convertir /ruta/a/documentos --engine auto --skip-existing

# Descripción de imágenes (opcional, requiere ANTHROPIC_API_KEY)
python -m pipeline captions --dry-run     # cuenta, no llama a la API
python -m pipeline captions --limit 3     # smoke test
python -m pipeline captions               # todo

# Fase 4 — control de calidad
python -m pipeline qc

# Todo de una
python -m pipeline todo /ruta/a/documentos --engine auto --captions
```

Los scripts originales siguen funcionando como wrappers: `inventario.py`,
`convertir_pymupdf.py`, `convertir_marker.sh`, `convertir_docx.sh`,
`convertir_pptx.py`, `caption_imagenes.py`, `audit_dedup_captions.py`,
`qc_inventario.py`.

### Salida

Todo queda en `output/<estructura de carpetas saneada>/<documento>/`, con
imágenes en `assets/` (o `media/` para pandoc; Marker las deja planas junto al
`.md`) y enlaces relativos al propio `.md` — la carpeta de cada documento es
autocontenida y portable.

La estructura relativa se calcula contra el **ancestro común del lote**. Si
mezclas documentos de árboles muy distintos (p. ej. `/Users/...` y `/mnt/...`),
el ancestro común es `/` y las rutas de salida salen profundas; usa
`--input-root /ruta/base` para fijarla explícitamente.

### Flags útiles

| Flag | Para qué |
|---|---|
| `--engine auto\|pymupdf\|marker` | Motor de PDF; `auto` decide por documento |
| `--skip-existing` | No reconvertir lo que ya tiene un `.md` no vacío — reanuda un lote interrumpido |
| `--out DIR` | Directorio de salida (default `output`) |
| `--input-root DIR` | Raíz para las rutas relativas de salida |

## Interfaz web

```bash
pip install -r requirements-web.txt
uvicorn webapp.main:app --host 0.0.0.0 --port 8000
```

Abrir `http://localhost:8000`. Permite subir varios documentos a la vez, elegir
motor, activar el captioning, seguir el progreso en vivo y descargar el
resultado como ZIP. Los trabajos se procesan en una cola en segundo plano y
sobreviven a un reinicio del servicio (se reanudan donde estaban).

| Variable | Default | Para qué |
|---|---|---|
| `DATA_DIR` | `data` | Dónde viven subidas, resultados y la base de trabajos |
| `WORKERS` | `1` | Trabajos simultáneos. Con Marker dejar en 1: uno ya satura la máquina |
| `ANTHROPIC_API_KEY` | — | Habilita el paso opcional de captioning |
| `MAX_UPLOAD_MB` | `500` | Tope por archivo |
| `RETENTION_DAYS` | `14` | Borra trabajos terminados más antiguos (0 = nunca) |
| `MARKER_MODE` | *(vacío)* | Vacío = Marker elige por dispositivo. `fast` o `balanced` para forzarlo |
| `MARKER_TIMEOUT_S` | `14400` | Corta un documento colgado en vez de bloquear el lote |
| `CAPTION_BACKEND` | `ollama` | `ollama` (local, nada sale) o `anthropic` (API externa) |
| `OLLAMA_HOST` | `http://localhost:11434` | En Docker: `http://host.docker.internal:11434` |
| `CAPTION_MODEL` | `qwen3-vl:32b` | Modelo de visión (o `claude-sonnet-5` con backend anthropic) |
| `CAPTION_WORKERS` | `2` (ollama) | Ollama atiende de a una; subirlo no acelera |
| `OLLAMA_TIMEOUT_S` | `300` | Un 30B tarda decenas de segundos por imagen |

> **La app no tiene autenticación.** Cualquiera que alcance el puerto puede
> subir documentos, ver los de los demás y borrarlos. Por eso el
> `docker-compose.yml` la publica solo en `127.0.0.1`. Para exponerla en la red
> o a internet, poner delante un proxy inverso que resuelva el acceso
> (nginx/Caddy con autenticación y TLS, o una VPN) — no basta con abrir el
> puerto.

## Despliegue con Docker

```bash
# Imagen completa (con Marker/OCR). Tarda: arrastra torch + CUDA (~9.7 GB
# de imagen final, medido en arm64).
docker compose up -d --build

# Imagen ligera, sin Marker ni torch (~1.3 GB): PDFs nativos, docx y pptx.
# Sin OCR: un PDF escaneado sale vacío.
docker build --build-arg WITH_MARKER=false -t doc2md-light .
```

Configuración por `.env` junto al `docker-compose.yml`:

```env
PORT=8000
WORKERS=1
# ANTHROPIC_API_KEY=sk-ant-...   # solo si quieres captioning de imágenes
```

El servicio se publica en `127.0.0.1:8000` del host. Para llegar desde otra
máquina, ponle delante un proxy inverso (que además es donde toca resolver
autenticación y TLS) en vez de abrir el puerto directamente.

Todo el estado (subidas, resultados, base de trabajos y **la caché de pesos del
modelo**) vive en el volumen `doc2md-data` montado en `/data`. El contenedor es
desechable; el volumen no. Sin ese volumen, Marker vuelve a descargar 2-4 GB de
pesos en cada recreación.

### GPU NVIDIA

Los wheels de torch que instala `marker-pdf` en linux/amd64 ya traen CUDA, así
que **la misma imagen sirve para CPU y GPU** — lo que cambia es el runtime:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Verificar que el contenedor ve la GPU:
docker compose exec app python -c "import torch; print(torch.cuda.is_available())"
```

Requiere el NVIDIA Container Toolkit en el host.

**Para aprovechar la GPU de verdad hace falta el modo `balanced`**, y ese modo
necesita el binario `llama-server`, que la imagen por defecto no trae:

```bash
docker build --build-arg WITH_MARKER=true --build-arg WITH_LLAMA_SERVER=true -t doc2md .
```

Compila llama.cpp en una etapa aparte (solo el binario llega a la imagen
final). Sin él, Marker corre en modo `fast` incluso con GPU presente, y forzar
`MARKER_MODE=balanced` falla con `llama-server binary not found`.

### CLI dentro del contenedor

```bash
docker compose run --rm app cli inventario /data/entrada
docker compose run --rm app cli convertir /data/entrada --engine auto --out /data/salida
```

## ¿Este equipo o un servidor con GPU?

Solo una etapa del pipeline es pesada, así que la decisión depende de cuántas
páginas caen en la ruta Marker — no de cuántos documentos hay:

| Etapa | Recurso | ¿Gana con GPU NVIDIA? |
|---|---|---|
| inventario / QC | CPU, I/O | No |
| pymupdf4llm, pandoc, pptx | CPU | No — segundos por documento |
| **Marker `--mode balanced`** | **VLM** | **Sí, mucho** |
| captioning | red / API de Claude | No — es latencia de red |

Con ~15 s/página de media en Apple Silicon (Metal), un MacBook Pro M3 Pro hace
~240 páginas/hora en Marker:

- **< 2.000 páginas Marker** → local, de noche. No hace falta servidor.
- **2.000–10.000** → local es viable en un fin de semana; una GPU rentada se
  paga sola en horas ahorradas.
- **> 10.000, o proceso recurrente** → servidor. En una L4/A10/4090 el camino
  torch hace batching real, que en Apple Silicon vía `llama-server` no existe
  (es un solo stream): entre 5x y 15x.

A la velocidad se suma la calidad: en CPU, Marker corre en modo `fast`
(detectores ligeros); el modo `balanced` — layout VLM + OCR de página completa,
que es lo que resuelve bien la multicolumna — es el default *en GPU*. Un
servidor con GPU no solo termina antes: produce mejor Markdown.

**Corre `python -m pipeline inventario` sobre el corpus real antes de decidir.**
Si sale mayormente `nativo`, todo esto es discutir sobre nada: pymupdf4llm lo
resuelve en minutos en cualquier portátil.

**Y antes de la velocidad, mira la sensibilidad de los documentos.** Si no
pueden salir de la organización, eso descarta tanto la GPU rentada en la nube
como el captioning vía API — y entonces "esperar toda la noche en el portátil"
es la opción correcta, no la lenta.

Nota de memoria: 18 GB unificados bastan para un stream de Marker (pesos 2-4 GB
+ torch + llama-server), pero van justos con Docker, IDEs y navegador abiertos.
La presión de memoria manda a swap y la degradación es brutal — cierra lo demás
antes de un lote largo.

## Problemas conocidos

- **pymupdf4llm 1.28.x**: con `write_images=True`, si la ruta de salida tiene
  espacios o paréntesis, el guardado de imágenes falla (`code=2: cannot open
  file`) por una inconsistencia entre la ruta saneada para el enlace Markdown y
  la real usada en `pix.save()`. Workaround aplicado en `pipeline/paths.py`
  (`slugify`): se sanean los nombres antes de pasarlos a la librería.
- **surya-ocr 0.22.x** (dependencia de marker-pdf, modo `balanced`): bug de
  gramática GBNF con `\d` que rompe el layout inference en Mac/CPU/MPS. Ver
  [datalab-to/surya#542](https://github.com/datalab-to/surya/issues/542) y
  `patches/fix_surya_grammar.py`.
- **marker-pdf 2.0.0**: si una caja de layout se recorta a área cero (bbox
  degenerado), Pillow revienta con `ValueError: cannot write empty image as
  JPEG` y aborta la conversión COMPLETA del documento, aunque llevara 20+
  minutos. Ver `patches/fix_marker_empty_image.py` (salta esa imagen puntual
  con un aviso en vez de abortar). Sin fix upstream conocido a la fecha.
- **`marker_single` anida su propio output**: crea
  `--output_dir/<nombre original>/<archivo>.md` en vez de
  `--output_dir/<archivo>.md`, y con el nombre *sin sanear*.
  `pipeline/converters.py` lo aplana automáticamente.
- **Multicolumna**: incluso con Marker, revisar manualmente documentos con
  layouts muy irregulares (outlines multinivel, tablas anidadas) — es el punto
  débil común a ambas herramientas.
- **PPTX con `.emf`/`.wmf`**: esos formatos (típicos de diagramas pegados desde
  Office) se extraen igual, pero se enlazan como adjunto y no como imagen —
  ningún visor de Markdown los renderiza.

## Procesamiento de imágenes

Las imágenes se extraen como archivos separados con enlaces relativos
(`assets/imagen.png`), nunca embebidas en base64 dentro del `.md`. Motivos:

- **Base64 inline infla el archivo ~33%** y vuelve los diffs de git
  ilegibles — malo para versionado y para indexar en un pipeline de RAG (los
  chunks de texto quedan contaminados con blobs enormes).
- **El costo en tokens de un modelo de visión depende de la RESOLUCIÓN de la
  imagen, no del encoding.** Claude tokeniza en parches: una imagen de
  1000×1000px cuesta lo mismo venga como base64 o como archivo. Base64 no ahorra
  ni cuesta tokens — es puramente un formato de transporte.
- Por lo tanto **el archivo separado es estrictamente mejor para
  almacenamiento/versionado**, y convertir a base64 (si cierta API solo acepta
  eso) es responsabilidad del consumidor en el momento de la llamada.

### Captioning de imágenes

Para que el contenido de diagramas/mapas/tablas-como-imagen sea *buscable por
texto* (un RAG de solo texto no puede "ver" una imagen), se genera una vez una
descripción corta de cada imagen y se inserta justo debajo.

Dos backends con la misma interfaz, vía `CAPTION_BACKEND`:

| Backend | Modelo por defecto | Privacidad | Velocidad |
|---|---|---|---|
| **`ollama`** (default) | `qwen3-vl:32b` | **Las imágenes no salen de la máquina** | Lento (decenas de s/imagen) |
| `anthropic` | `claude-sonnet-5` | Cada imagen viaja a un tercero | Rápido (minutos por lote) |

```bash
python -m pipeline captions --limit 3    # smoke test
python -m pipeline captions              # todo
python -m pipeline captions-audit        # verificar 1 caption por imagen

# Forzar backend o modelo puntualmente
python -m pipeline captions --backend ollama --model qwen3-vl:30b
python -m pipeline captions --ollama-host http://otra-maquina:11434
```

Antes de procesar el lote se hace un **preflight**: si el servidor no responde o
el modelo no está descargado, falla de inmediato con un mensaje accionable en
vez de acumular un error por imagen a mitad de un lote de horas.

#### Ollama en el servidor con Docker

Un detalle que rompe el despliegue si se pasa por alto: **Ollama escucha por
defecto solo en `127.0.0.1`**, así que desde el contenedor —que llega por la
interfaz de Docker— la conexión es rechazada. Hay que abrirlo al host:

```bash
sudo systemctl edit ollama
```
```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```
```bash
sudo systemctl restart ollama

# Verificar desde el contenedor (debe listar tus modelos):
docker compose exec app sh -c 'curl -s "$OLLAMA_HOST/api/tags"'
```

Si prefieres no exponer Ollama ni siquiera a la red local, la alternativa es
`network_mode: host` en el servicio, y entonces `OLLAMA_HOST=http://localhost:11434`.

El `docker-compose.yml` ya trae `extra_hosts: host.docker.internal:host-gateway`,
que en Linux es necesario (en Mac y Windows Docker lo define solo).

- **Es reanudable**: las imágenes que ya tienen caption se detectan y se saltan
  sin llamar al modelo. Reejecutar tras un fallo a mitad de lote no repite
  trabajo ni duplica descripciones. (`captions-dedup` sigue disponible para
  limpiar duplicados heredados de corridas anteriores.)
- **Local (`ollama`, por defecto)**: sin costo, sin clave y sin que las imágenes
  salgan de la máquina. `qwen3-vl:32b` (21 GB) o `:30b` (20 GB) si la GPU tiene
  24 GB; `qwen3-vl:8b` (6.1 GB) o `:4b` (3.3 GB) con menos memoria — un equipo
  de 18 GB no puede con los de 30B+. A cambio es lento: decenas de segundos por
  imagen, frente a minutos por lote completo vía API.
- **API (`anthropic`)**: minutos y ~$1-2 para varios cientos de imágenes, con
  `ANTHROPIC_API_KEY` y `CAPTION_MODEL=claude-sonnet-5`. Mejor calidad y mucho
  más rápido, pero cada imagen viaja a un tercero — no usar con documentos
  sensibles.
- **Cuidado con la concurrencia**: si varios hilos escriben el mismo `.md` sin
  lock hay *lost updates* (captions que desaparecen sin error visible).
  `pipeline/captions.py` usa un lock por archivo — no quitarlo.

Sources:
- [markitdown issue #2049 — extraer imágenes como archivos separados](https://github.com/microsoft/markitdown/issues/2049)
- [Building a Multimodal LLM Application with PyMuPDF4LLM](https://artifex.com/blog/building-a-multimodal-llm-application-with-pymupdf4llm)
- [Claude Vision docs — cálculo de tokens por imagen](https://platform.claude.com/docs/en/build-with-claude/vision)
- [Multimodal RAG: Retrieving from Images, PDFs, and Tables](https://tensoria.fr/en/blog/multimodal-rag-images-pdfs-tables)
