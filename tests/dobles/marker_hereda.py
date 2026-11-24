# Doble de `marker_single` que deja un nieto en su propia sesión HEREDANDO el
# pipe de salida y vivo tras terminar (como un servidor con keep-alive mal
# lanzado). La lectura del padre nunca verá EOF: no debe colgarse por ello.
import os
import subprocess
import sys
from pathlib import Path

codigo = "import time\nprint('ruido del nieto', flush=True)\nwhile True: time.sleep(1)\n"
nieto = subprocess.Popen([sys.executable, "-c", codigo, __file__], start_new_session=True)
with open(os.environ["INH_PID"], "w") as f:
    f.write(str(nieto.pid))

src = sys.argv[1]
out = sys.argv[sys.argv.index("--output_dir") + 1]
nombre = Path(src).stem
d = Path(out) / nombre
d.mkdir(parents=True, exist_ok=True)
(d / f"{nombre}.md").write_text("# x\n")
print("hecho", flush=True)
