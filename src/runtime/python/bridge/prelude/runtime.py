def _jetstream_exec_source(source):
    exec(source, globals())


def _jetstream_load_compute(source, _exec_source=_jetstream_exec_source):
    _exec_source(source)
    function = globals().get("compute")
    if not callable(function):
        raise RuntimeError("Python source must define a callable compute() function.")
    return function


def _jetstream_bind_compute(function, ctx):
    def _runner():
        return function(ctx)

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
