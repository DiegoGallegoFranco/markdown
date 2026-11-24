# Doble de `llama-server`: no hace nada, solo vive hasta que lo maten. Se
# llama `llama-server` en el PATH para que `ps` lo identifique como tal.
import time

while True:
    time.sleep(1)
