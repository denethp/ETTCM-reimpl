"""Copy everything a script prints to <project>/results_read.txt (cleared on each run),
so results can be read back without copy-pasting from the terminal."""
import datetime
import os
import sys

DEFAULT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results_read.txt"))


class _Tee:
    def __init__(self, stream, f):
        self.stream, self.f = stream, f

    def write(self, s):
        self.stream.write(s)
        if not s.startswith("\r"):          # progress lines stay on screen only
            self.f.write(s)
            self.f.flush()
        return len(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()


def tee_output(path=DEFAULT):
    """Start copying stdout and stderr to `path` (overwritten)."""
    f = open(path, "w", encoding="utf-8")
    f.write(f"# {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  $ python {' '.join(sys.argv)}\n")
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)
    print(f"(output also saved to {path})")
    return path
