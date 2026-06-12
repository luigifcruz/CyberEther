def _jetstream_exec_source(source):
    def _exec():
        exec(source, globals())

    return _jetstream_with_console(_exec)


def _jetstream_load_compute(source, _exec_source=_jetstream_exec_source):
    _exec_source(source)
    function = globals().get("compute")
    if not callable(function):
        raise RuntimeError("Python source must define a callable compute() function.")
    return function


def _jetstream_call_compute(function, ctx):
    return _jetstream_with_console(function, ctx)


def _jetstream_bind_compute(function, ctx, _call_compute=_jetstream_call_compute):
    def _runner():
        return _call_compute(function, ctx)

    return _runner
