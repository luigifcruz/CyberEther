def _jetstream_classify_attribute(value):
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return 1
    if isinstance(value, float):
        return 2
    if isinstance(value, str):
        return 3
    if isinstance(value, dict):
        return 4
    if isinstance(value, (list, tuple)):
        return 5
    return -1
