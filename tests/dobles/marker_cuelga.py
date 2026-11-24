# Doble de `marker_single` que se cuelga, con un hijo que IGNORA SIGTERM.
# Registra "<pid_marker> <pid_hijo>" en HANG_PIDS. El hijo lleva la ruta de
# este script en su argv para que el teardown del test pueda encontrarlo.
import os
import subprocess
import sys
import time

codigo = ("import signal, time\n"
          "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
          "while True: time.sleep(1)\n")
hijo = subprocess.Popen([sys.executable, "-c", codigo, __file__])
with open(os.environ["HANG_PIDS"], "w") as f:
    f.write(f"{os.getpid()} {hijo.pid}\n")
print("colgado", flush=True)
while True:
    time.sleep(1)
