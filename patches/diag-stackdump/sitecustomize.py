"""Dump every thread's Python stack on SIGUSR1, in every vLLM process.

A DIAGNOSTIC, not a submission. It changes no kernel and no arithmetic. Its only
purpose is to answer "where is the process stuck?" for the intermittent TP2
deadlock, where both ranks spin at 100% CPU with the GPUs at idle wattage.

py-spy is the usual tool and it does not work on this fleet: it is not installed,
and /proc/sys/kernel/yama/ptrace_scope is 1, so a process may only be traced by
its own descendants. Raising that needs root. This needs neither -- the arena
already runs code at interpreter startup in every engine and worker process, so
the process is made to dump itself.

Use:
    kill -USR1 <pid>          # on each rank, on each node
    cat /tmp/arena-stack-<pid>.txt

faulthandler writes from a signal handler, so it works even while the main
thread is blocked in a C-level collective wait -- which is exactly the state that
needs describing, and the state a pure-Python handler could never report.
"""

import faulthandler
import os
import signal
import sys

_path = f"/tmp/arena-stack-{os.getpid()}.txt"
try:
    # Held open for the process lifetime on purpose: the handler must not
    # allocate or open files at signal time.
    _fh = open(_path, "a", buffering=1)
    faulthandler.register(signal.SIGUSR1, file=_fh, all_threads=True, chain=False)
    print(f"arena diag-stackdump: SIGUSR1 -> {_path}", file=sys.stderr)
except Exception as _e:  # noqa: BLE001 - a diagnostic must never break the run
    print(f"arena diag-stackdump: not installed ({_e})", file=sys.stderr)
