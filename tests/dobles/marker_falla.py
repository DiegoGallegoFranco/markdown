# Doble de `marker_single` que falla con rc=3 tras escribir bastante salida,
# una barra tqdm a medias y un traceback al final.
import sys

for i in range(10):
    print(f"linea {i}", flush=True)
sys.stderr.write("\r 50%|##| 1/2 [00:01<00:01]\r")
sys.stderr.write("Traceback (most recent call last):\nRuntimeError: boom\n")
sys.stderr.flush()
sys.exit(3)
