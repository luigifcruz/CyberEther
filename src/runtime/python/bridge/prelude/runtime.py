SKIP = object()


class _JetstreamYield:
    __slots__ = ("delay",)

    def __init__(self, delay=float("inf")):
        self.delay = delay

    def __call__(self, seconds):
        seconds = float(seconds)
        if not seconds >= 0.0:
            raise ValueError("YIELD() takes a delay of zero or more seconds.")
        return _JetstreamYield(seconds)


YIELD = _JetstreamYield()


def _jetstream_check_reserved(skip, yield_):
    if globals().get("SKIP") is not skip:
        globals()["SKIP"] = skip
        raise RuntimeError("Python source cannot redefine the reserved SKIP sentinel.")

    if globals().get("YIELD") is not yield_:
        globals()["YIELD"] = yield_
        raise RuntimeError("Python source cannot redefine the reserved YIELD sentinel.")


def _jetstream_exec_source(source, source_file):
    source = source.decode("utf-8")
    if source_file:
        globals()["__file__"] = source_file
    exec(compile(source, source_file or "<string>", "exec"), globals())


def _jetstream_load_compute(
    source,
    source_file="",
    _exec_source=_jetstream_exec_source,
    _check_reserved=_jetstream_check_reserved,
    _skip=SKIP,
    _yield=YIELD,
):
    _exec_source(source, source_file)
    _check_reserved(_skip, _yield)
    function = globals().get("compute")
    if not callable(function):
        raise RuntimeError("Python source must define a callable compute() function.")
    return function


def _jetstream_bind_compute(
    function,
    ctx,
    _check_reserved=_jetstream_check_reserved,
    _skip=SKIP,
    _yield=YIELD,
    _yield_type=_JetstreamYield,
):
    def _runner():
        completed = False
        try:
            result = function(ctx)
            completed = True
        finally:
            if not completed:
                globals()["SKIP"] = _skip
                globals()["YIELD"] = _yield

        _check_reserved(_skip, _yield)
        if result is _skip:
            return (1, 0.0)
        if isinstance(result, _yield_type):
            return (2, result.delay)
        return (0, 0.0)

    return _runner


def _jetstream_shutdown():
    try:
        cleanup = globals().get("cleanup")
        if callable(cleanup):
            cleanup()
    finally:
        import gc as _jetstream_cleanup_gc

        namespace = globals()
        for key in list(namespace):
            if key.startswith("__") or key.lower().startswith("_jetstream_"):
                continue
            namespace[key] = None
        _jetstream_cleanup_gc.collect()


def _jetstream_cleanup_multiprocessing():
    import sys as _jetstream_cleanup_sys

    if not any(name == "multiprocessing" or name.startswith("multiprocessing.")
               for name in _jetstream_cleanup_sys.modules):
        return

    try:
        import gc as _jetstream_cleanup_gc
        import multiprocessing.util as _jetstream_mp_util
    except Exception:
        return

    for _ in range(2):
        try:
            _jetstream_mp_util._run_finalizers(0)
        except Exception:
            pass
        try:
            _jetstream_cleanup_gc.collect()
        except Exception:
            pass

    try:
        _jetstream_mp_util._run_finalizers()
    except Exception:
        pass

    _jetstream_forkserver = _jetstream_cleanup_sys.modules.get("multiprocessing.forkserver")
    if _jetstream_forkserver is not None:
        try:
            _jetstream_forkserver._forkserver._stop()
        except Exception:
            pass

    _jetstream_resource_tracker = _jetstream_cleanup_sys.modules.get("multiprocessing.resource_tracker")
    if _jetstream_resource_tracker is not None:
        try:
            _jetstream_resource_tracker._resource_tracker._stop()
        except Exception:
            pass


def _jetstream_shutdown_runtime():
    return _jetstream_cleanup_multiprocessing()
