class _JetstreamEnvironment(dict):
    def __init__(self):
        super().__init__()
        self._jetstream_dirty = set()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._jetstream_dirty.add(key)

    def __delitem__(self, key):
        super().__delitem__(key)
        self._jetstream_dirty.add(key)

    def pop(self, key, *args):
        result = dict.pop(self, key, *args)
        self._jetstream_dirty.add(key)
        return result

    def popitem(self):
        key, value = dict.popitem(self)
        self._jetstream_dirty.add(key)
        return key, value

    def clear(self):
        self._jetstream_dirty.update(dict.keys(self))
        dict.clear(self)

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return dict.__getitem__(self, key)

    def update(self, *args, **kwargs):
        for arg in args:
            if hasattr(arg, "keys"):
                for key in arg.keys():
                    self[key] = arg[key]
            else:
                for key, value in arg:
                    self[key] = value
        for key, value in kwargs.items():
            self[key] = value


class _JetstreamTrackedDict(dict):
    def __init__(self, mark):
        super().__init__()
        self._jetstream_mark = mark

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        self._jetstream_mark()

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        self._jetstream_mark()

    def pop(self, *args):
        result = dict.pop(self, *args)
        self._jetstream_mark()
        return result

    def popitem(self):
        result = dict.popitem(self)
        self._jetstream_mark()
        return result

    def clear(self):
        dict.clear(self)
        self._jetstream_mark()

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return dict.__getitem__(self, key)

    def update(self, *args, **kwargs):
        dict.update(self, *args, **kwargs)
        self._jetstream_mark()


def _jetstream_wrap_environment_value(value, mark):
    if isinstance(value, dict):
        wrapped = _JetstreamTrackedDict(mark)
        for key, entry in value.items():
            dict.__setitem__(wrapped, key, _jetstream_wrap_environment_value(entry, mark))
        return wrapped
    if isinstance(value, (list, tuple)):
        return tuple(_jetstream_wrap_environment_value(entry, mark) for entry in value)
    return value


def _jetstream_track_environment(env, keys=None):
    dirty = env._jetstream_dirty
    if keys is None:
        keys = list(dict.keys(env))
    for key in keys:
        if not dict.__contains__(env, key):
            continue
        def _jetstream_mark(_key=key, _dirty=dirty):
            _dirty.add(_key)
        dict.__setitem__(
            env,
            key,
            _jetstream_wrap_environment_value(dict.__getitem__(env, key), _jetstream_mark),
        )


def _jetstream_create_environment():
    return _JetstreamEnvironment()


def _jetstream_consume_dirty_environment(env):
    dirty = getattr(env, "_jetstream_dirty", None)
    if not dirty:
        return ()
    keys = tuple(dirty)
    dirty.clear()
    return keys
