"""Keep Windows from sleeping while generations are queued (an overnight batch must finish).

Only the system is kept awake; the screen may still turn off. The request belongs to the
calling thread, so the job worker calls this itself, and it ends when the queue is empty.
"""

import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake(on: bool) -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except (OSError, AttributeError):
        pass  # not critical: worst case the PC sleeps as it normally would
