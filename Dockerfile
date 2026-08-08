# Imagen del pipeline + interfaz web.
#
#   docker build -t doc2md .                          # con Marker (OCR real, ~9.7 GB)
#   docker build -t doc2md-light --build-arg WITH_MARKER=false .   # sin Marker (~1.3 GB)
#   docker build -t doc2md --build-arg WITH_LLAMA_SERVER=true .    # + modo balanced
#
# En linux/amd64, `pip install marker-pdf` trae los wheels de torch con soporte
# CUDA, así que la MISMA imagen corre en CPU y en GPU NVIDIA: lo que cambia es
# arrancarla con el runtime de NVIDIA (ver docker-compose.gpu.yml).
#
# Sobre los modos de Marker: `fast` usa detectores CPU ligeros; `balanced` usa
# el modelo de layout VLM + OCR de página completa y da mejor calidad, pero
# necesita el binario `llama-server` de llama.cpp. Marker elige solo por
# dispositivo (fast en CPU, balanced en GPU), así que el binario solo hace
# falta si vas a correr en GPU o forzar MARKER_MODE=balanced: para eso está
# WITH_LLAMA_SERVER, que lo compila en una etapa aparte.

# --------------------------------------------------------------------- llama.cpp
FROM python:3.12-slim AS llama-builder
ARG WITH_LLAMA_SERVER=false
# Pineado: una release concreta, no la rama, para que el build sea reproducible.
ARG LLAMA_CPP_REF=b6193
RUN if [ "$WITH_LLAMA_SERVER" = "true" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            build-essential cmake git libcurl4-openssl-dev ca-certificates \
        && git clone --depth 1 --branch "$LLAMA_CPP_REF" \
            https://github.com/ggml-org/llama.cpp /src/llama.cpp \
        && cmake -S /src/llama.cpp -B /src/build \
            -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=ON \
        && cmake --build /src/build --target llama-server -j "$(nproc)" \
        && mkdir -p /out && cp /src/build/bin/llama-server /out/ ; \
    else \
        mkdir -p /out ; \
    fi

# --------------------------------------------------------------------- imagen final
FROM python:3.12-slim

ARG WITH_MARKER=true
ARG WITH_LLAMA_SERVER=false

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data \
    # Los pesos de los modelos (~2-4 GB) se descargan una vez y viven en el
    # volumen: sin esto se re-descargan en cada recreación del contenedor.
    HF_HOME=/data/hf-cache \
    TORCH_HOME=/data/torch-cache

# pandoc: conversión de .docx. libgl1/libglib: dependencias nativas de
# opencv/pillow que arrastra surya. tini: reaping de procesos hijos (marker_single).
RUN apt-get update && apt-get install -y --no-install-recommends \
        pandoc \
        libgl1 \
        libglib2.0-0 \
        tini \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-light.txt requirements-web.txt ./
RUN if [ "$WITH_MARKER" = "true" ]; then \
        pip install -r requirements.txt ; \
    else \
        pip install -r requirements-light.txt ; \
    fi \
    && pip install -r requirements-web.txt

COPY pipeline/ ./pipeline/
COPY webapp/ ./webapp/
COPY patches/ ./patches/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY *.py *.sh ./

# Los parches se aplican en la imagen, no en runtime, y FALLAN el build si el
# patrón esperado no está: sin ellos Marker cae a un modo degradado sin avisar
# (surya #542) y un bbox degenerado aborta la conversión completa de un
# documento tras 20+ minutos de proceso.
RUN if [ "$WITH_MARKER" = "true" ]; then \
        python patches/fix_surya_grammar.py && \
        python patches/fix_marker_empty_image.py ; \
    fi

# Marker escribe por defecto DENTRO de site-packages (la fuente, resultados
# intermedios, datos de debug). Con un usuario no-root eso es un
# PermissionError en la PRIMERA conversión — no al arrancar, así que pasa
# desapercibido hasta que alguien sube un documento. Se redirige todo a rutas
# escribibles.
ENV FONT_DIR=/opt/marker-assets/fonts \
    FONT_PATH=/opt/marker-assets/fonts/GoNotoCurrent-Regular.ttf \
    OUTPUT_DIR=/data/marker-tmp \
    DEBUG_DATA_FOLDER=/data/marker-debug

# La fuente se descarga en el build (con red y como root) en vez de en la
# primera conversión: así el contenedor no necesita salida a internet ni
# permisos de escritura para convertir. El `test -s` falla el build si marker
# cambia el nombre de la fuente y FONT_PATH deja de cuadrar, en vez de dejarlo
# reventar en runtime.
RUN if [ "$WITH_MARKER" = "true" ]; then \
        mkdir -p "$FONT_DIR" && \
        python -c "from marker.util import download_font; download_font()" && \
        test -s "$FONT_PATH" ; \
    fi

# Vacío si WITH_LLAMA_SERVER=false; con el binario si se pidió el modo balanced.
# Va al final para no invalidar la caché del pip install al cambiar de opción.
COPY --from=llama-builder /out/ /usr/local/bin/

RUN chmod +x /usr/local/bin/entrypoint.sh /app/*.sh \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /data /opt/marker-assets \
    && chown -R app:app /data /app /opt/marker-assets
USER app

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["web"]
