# Imagen del pipeline + interfaz web.
#
#   docker build -t doc2md .                          # con Marker (OCR real, ~9.7 GB)
#   docker build -t doc2md-light --build-arg WITH_MARKER=false .   # sin Marker (~1.3 GB)
#   docker build -t doc2md --build-arg WITH_LLAMA_SERVER=true \
#       --build-arg CUDA_ARCHITECTURES=75 .                        # + modo balanced (GPU)
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
#
# Se compila CON CUDA: en un servidor con GPU, Marker elige `balanced` solo, y
# ese modo corre su VLM sobre llama.cpp. Con un llama-server CPU-only la parte
# torch iría a la GPU pero el VLM se quedaría en CPU, que es la etapa lenta —
# la GPU quedaría medio ociosa y el documento tardaría casi lo mismo.
#
# Dos etapas y se elige una por nombre (`llama-true` / `llama-false`) en vez de
# un if dentro del RUN: así la imagen de CUDA (~4 GB) solo se descarga cuando
# realmente se pide el binario. BuildKit no construye las etapas que nadie usa.
ARG WITH_LLAMA_SERVER=false
# ubuntu22.04 a propósito (glibc 2.35): el binario tiene que correr en la imagen
# final, que es Debian slim. Compilar contra una glibc MÁS NUEVA que la del
# destino produce un binario que no arranca.
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu22.04

FROM ${CUDA_IMAGE} AS llama-true
# Pineado: una release concreta, no la rama, para que el build sea reproducible.
ARG LLAMA_CPP_REF=b6193
# Arquitecturas CUDA a compilar. 75 = Turing (Quadro RTX 8000, RTX 20xx, T4).
# 80 = A100, 86 = RTX 30xx/A6000, 89 = RTX 40xx/L4. Cada una añade minutos de
# compilación, así que por defecto va solo la del servidor de destino.
ARG CUDA_ARCHITECTURES=75
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git libcurl4-openssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# BUILD_SHARED_LIBS=OFF es obligatorio, no una optimización: por defecto
# llama.cpp deja llama-server enlazado contra libllama.so y libggml*.so, y a la
# imagen final solo llega el ejecutable. Con las .so fuera, el binario existe
# pero muere al arrancar ("error while loading shared libraries"), y como en CPU
# Marker nunca lo invoca el fallo solo aparece al encender la GPU.
RUN git clone --depth 1 --branch "$LLAMA_CPP_REF" \
        https://github.com/ggml-org/llama.cpp /src/llama.cpp \
    && cmake -S /src/llama.cpp -B /src/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_CURL=ON \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
    && cmake --build /src/build --target llama-server -j "$(nproc)"
# ggml queda estático, pero CUDA no: cudart/cublas siguen siendo dinámicas y la
# imagen final no las tiene (libcuda.so, la del driver, la inyecta el NVIDIA
# Container Toolkit en runtime). Viajan junto al binario.
RUN mkdir -p /out/bin /out/lib \
    && cp /src/build/bin/llama-server /out/bin/ \
    && for lib in libcudart libcublas libcublasLt; do \
           cp -a /usr/local/cuda/lib64/$lib.so.* /out/lib/ ; \
       done \
    && touch /out/bin/.keep /out/lib/.keep

# Sin llama-server: directorios vacíos para que los COPY de abajo no fallen.
FROM python:3.12-slim AS llama-false
RUN mkdir -p /out/bin /out/lib && touch /out/bin/.keep /out/lib/.keep

FROM llama-${WITH_LLAMA_SERVER} AS llama-builder

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
# libgomp1: OpenMP, que ggml enlaza en el llama-server compilado aparte.
RUN apt-get update && apt-get install -y --no-install-recommends \
        pandoc \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
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
COPY --from=llama-builder /out/bin/ /usr/local/bin/
COPY --from=llama-builder /out/lib/ /usr/local/lib/llama-cuda/
# ldconfig en vez de LD_LIBRARY_PATH: la variable la heredarían también los
# procesos hijos de Marker, y sus propias librerías CUDA (las que traen los
# wheels de torch) pasarían a resolverse contra estas copias.
RUN rm -f /usr/local/bin/.keep /usr/local/lib/llama-cuda/.keep \
    && echo /usr/local/lib/llama-cuda > /etc/ld.so.conf.d/llama-cuda.conf \
    && ldconfig \
    # Falla el build si al binario le falta alguna .so, en vez de dejar que
    # reviente en la primera conversión en GPU (en CPU Marker no lo invoca
    # nunca, así que el fallo puede dormir meses). `ldd` resuelve enlaces sin
    # ejecutar nada: no hace falta GPU en la máquina que construye.
    #
    # libcuda.so.1 se excluye a propósito: es la librería del DRIVER, no viene
    # en ninguna imagen y la monta el NVIDIA Container Toolkit al arrancar el
    # contenedor. En build siempre falta, y eso es correcto.
    && if [ "$WITH_LLAMA_SERVER" = "true" ]; then \
           faltan="$(ldd /usr/local/bin/llama-server \
                     | grep 'not found' \
                     | grep -v 'libcuda\.so\.1' || true)" ; \
           if [ -n "$faltan" ]; then \
               echo "llama-server no resuelve sus librerías:" >&2 ; \
               echo "$faltan" >&2 ; \
               exit 1 ; \
           fi ; \
       fi

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
