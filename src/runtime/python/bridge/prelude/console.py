import sys as _jetstream_sys
import threading as _jetstream_threading

_JETSTREAM_CONSOLE_MAX_LINES = 256


class _JetstreamConsole:
    def __init__(self):
        self._lock = _jetstream_threading.Lock()
        self._lines = []
        self._pending = ""

    def write(self, text):
        text = str(text)
        with self._lock:
            self._pending += text
            lines = self._pending.split("\n")
            self._pending = lines.pop()
            self._lines.extend(line.rstrip("\r") for line in lines)
            overflow = len(self._lines) - _JETSTREAM_CONSOLE_MAX_LINES
            if overflow > 0:
                del self._lines[:overflow]
        return len(text)

    def flush(self):
        with self._lock:
            self._flush_pending()

    def snapshot(self):
        with self._lock:
            self._flush_pending()
            return list(self._lines)

    def _flush_pending(self):
        if self._pending:
            self._lines.append(self._pending.rstrip("\r"))
            self._pending = ""


_jetstream_console = _JetstreamConsole()


class _JetstreamConsoleDispatcher:
    _jetstream_console_dispatcher = True

    def __init__(self, fallback):
        self._fallback = fallback

    def write(self, text):
        return self._route().write(text)

    def flush(self):
        self._route().flush()

    def __getattr__(self, name):
        return getattr(self._fallback, name)

    def _route(self):
        try:
            frame = _jetstream_sys._getframe(2)
        except ValueError:
            frame = None
        while frame is not None:
            console = frame.f_globals.get("_jetstream_console")
            if console is not None:
                return console
            frame = frame.f_back
        return self._fallback


def _jetstream_install_console_dispatcher():
    if not getattr(_jetstream_sys.stdout, "_jetstream_console_dispatcher", False):
        _jetstream_sys.stdout = _JetstreamConsoleDispatcher(_jetstream_sys.stdout)
    if not getattr(_jetstream_sys.stderr, "_jetstream_console_dispatcher", False):
        _jetstream_sys.stderr = _JetstreamConsoleDispatcher(_jetstream_sys.stderr)


_jetstream_install_console_dispatcher()


def _jetstream_console_snapshot():
    return _jetstream_console.snapshot()


def _jetstream_format_exception(exc_type, exc_value, exc_traceback):
    import traceback as _jetstream_traceback
    return "".join(
        _jetstream_traceback.format_exception(exc_type, exc_value, exc_traceback)
    ).rstrip("\n")
