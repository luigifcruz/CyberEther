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
_jetstream_console_bypass = _jetstream_threading.local()


class _JetstreamConsoleDispatcher:
    _jetstream_console_dispatcher = True

    def __init__(self, fallback):
        self._fallback = fallback

    def write(self, text):
        return self._route().write(text)

    def writelines(self, lines):
        target = self._route()
        for line in lines:
            target.write(line)

    def flush(self):
        self._route().flush()

    def isatty(self):
        # TODO: Return True once the console UI renders ANSI colors.
        return False

    def __getattr__(self, name):
        if name == "_fallback":
            raise AttributeError(name)
        return getattr(self._fallback, name)

    def _route(self):
        if getattr(_jetstream_console_bypass, "active", False):
            return self._fallback
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


def _jetstream_install_thread_excepthook():
    previous = getattr(_jetstream_threading, "excepthook", None)
    if previous is None or getattr(previous, "_jetstream_console_excepthook", False):
        return

    def _previous_unrouted(args):
        _jetstream_console_bypass.active = True
        try:
            previous(args)
        finally:
            _jetstream_console_bypass.active = False

    def _jetstream_thread_excepthook(args):
        if args.exc_type is SystemExit:
            return
        try:
            console = None
            traceback = args.exc_traceback
            while traceback is not None:
                console = traceback.tb_frame.f_globals.get("_jetstream_console", console)
                traceback = traceback.tb_next
            if console is None:
                _previous_unrouted(args)
                return
            name = getattr(args.thread, "name", "<unknown>")
            console.write(f"Exception in thread {name}:\n")
            console.write(_jetstream_format_exception(args.exc_type, args.exc_value, args.exc_traceback) + "\n")
        except Exception:
            _previous_unrouted(args)

    _jetstream_thread_excepthook._jetstream_console_excepthook = True
    _jetstream_threading.excepthook = _jetstream_thread_excepthook


_jetstream_install_thread_excepthook()
