import sys as _jetstream_sys


class _JetstreamConsole:
    def __init__(self):
        self._lines = []
        self._pending = ""

    def write(self, text):
        text = str(text)
        self._pending += text
        lines = self._pending.split("\n")
        self._pending = lines.pop()
        self._lines.extend(line.rstrip("\r") for line in lines)
        return len(text)

    def flush(self):
        if self._pending:
            self._lines.append(self._pending.rstrip("\r"))
            self._pending = ""

    def snapshot(self):
        self.flush()
        return list(self._lines)


_jetstream_console = _JetstreamConsole()


def _jetstream_with_console(function, *args):
    previous_stdout = _jetstream_sys.stdout
    previous_stderr = _jetstream_sys.stderr
    _jetstream_sys.stdout = _jetstream_console
    _jetstream_sys.stderr = _jetstream_console
    try:
        return function(*args)
    finally:
        _jetstream_sys.stdout = previous_stdout
        _jetstream_sys.stderr = previous_stderr


def _jetstream_console_snapshot():
    return _jetstream_console.snapshot()
