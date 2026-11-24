# Doble de `marker_single` (marker-pdf 2.0.0 + surya-ocr 0.22.1).
#
# Imita lo que importa al pipeline:
#   - surya: se engancha al llama-server del sentinel (FAKE_SENTINEL) si está
#     vivo, o arranca uno; al salir lo mata salvo SURYA_INFERENCE_KEEP_ALIVE=true.
#     El servidor va con stdout/stderr a DEVNULL (el real escribe a un log): si
#     heredara el pipe, la lectura del padre no vería EOF.
#   - salida: `<output_dir>/<nombre SIN sanear>/<nombre>.md` + una imagen +
#     `<nombre>_meta.json` (que NO es una imagen).
#   - progreso tqdm por stderr, redibujado con \r y sin \n.
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

src = sys.argv[1]
out = sys.argv[sys.argv.index("--output_dir") + 1]

if os.environ.get("FAKE_ENV_LOG"):
    with open(os.environ["FAKE_ENV_LOG"], "a") as f:
        f.write(json.dumps({
            "backend": os.environ.get("SURYA_INFERENCE_BACKEND"),
            "keep_alive": os.environ.get("SURYA_INFERENCE_KEEP_ALIVE"),
            "argv": sys.argv[1:],
        }) + "\n")


def vivo(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


pid = None
lanzado = False
if os.environ.get("FAKE_SENTINEL"):
    sent = Path(os.environ["FAKE_SENTINEL"])
    if sent.exists():
        p = json.loads(sent.read_text())["pid"]
        if vivo(p):
            pid = p
    if pid is None:
        servidor = Path(__file__).with_name("llama-server")
        pid = subprocess.Popen([str(servidor)], start_new_session=True,
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL).pid
        sent.write_text(json.dumps({"pid": pid, "port": 1}))
        with open(os.environ["FAKE_SPAWNS"], "a") as f:
            f.write(f"{pid}\n")
        lanzado = True

print("Loading models", flush=True)
pasos = int(os.environ.get("FAKE_PASOS", "20"))
for i in range(pasos + 1):
    pct = i * 100 // pasos
    sys.stderr.write(f"\rRecognizing:  {pct}%|{'#' * (pct // 10)}| {i}/{pasos} [00:01<00:01, 5it/s]")
    sys.stderr.flush()
sys.stderr.write("\n")
print("Saved markdown", flush=True)

nombre = Path(src).stem
d = Path(out) / nombre
d.mkdir(parents=True, exist_ok=True)
(d / f"{nombre}.md").write_text("# Título\n\nTexto.\n\n![](_page_0_Picture_1.jpeg)\n")
(d / "_page_0_Picture_1.jpeg").write_bytes(b"\xff\xd8\xff\xe0fake")
(d / f"{nombre}_meta.json").write_text("{}")

if lanzado and os.environ.get("SURYA_INFERENCE_KEEP_ALIVE") != "true":
    os.kill(pid, signal.SIGTERM)
    Path(os.environ["FAKE_SENTINEL"]).unlink()
