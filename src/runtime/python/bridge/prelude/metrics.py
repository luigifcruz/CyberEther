_JETSTREAM_METRICS_SUBSCRIBE_ALL = ".jetstream.metrics.subscribe_all"


class _JetstreamMetrics(dict):
    def __init__(self):
        super().__init__()
        self._jetstream_requests = set()
        self._jetstream_subscribe_all = False

    def __missing__(self, key):
        self._jetstream_requests.add(key)
        return {}

    def get(self, key, default=None):
        if key not in self:
            self._jetstream_requests.add(key)
            return {} if default is None else default
        return dict.__getitem__(self, key)

    def subscribe_all(self):
        self._jetstream_subscribe_all = True


def _jetstream_create_metrics():
    return _JetstreamMetrics()


def _jetstream_consume_metrics_requests(metrics):
    requests = getattr(metrics, "_jetstream_requests", None)
    keys = tuple(requests) if requests else ()
    if requests:
        requests.clear()
    if getattr(metrics, "_jetstream_subscribe_all", False):
        metrics._jetstream_subscribe_all = False
        return (_JETSTREAM_METRICS_SUBSCRIBE_ALL,) + keys
    return keys
