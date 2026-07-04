_jetstream_value_converter_cache = []


def _jetstream_value_converters():
    if _jetstream_value_converter_cache:
        return _jetstream_value_converter_cache[0]

    try:
        import numpy as np
    except Exception:
        import struct

        def _tuple(fmt, itemsize):
            def convert(buffer):
                count = len(buffer) // itemsize
                return struct.unpack(f"={count}{fmt}", buffer)
            return convert

        def _complex_tuple(fmt, itemsize):
            def convert(buffer):
                count = (len(buffer) // itemsize) * 2
                flat = struct.unpack(f"={count}{fmt}", buffer)
                return tuple(complex(flat[i], flat[i + 1]) for i in range(0, len(flat), 2))
            return convert

        table = {
            "VU64": _tuple("Q", 8),
            "VF32": _tuple("f", 4),
            "VF64": _tuple("d", 8),
            "VCF32": _complex_tuple("f", 8),
            "VCF64": _complex_tuple("d", 16),
        }
        _jetstream_value_converter_cache.append(table)
        return table

    def _array(dtype):
        def convert(buffer):
            return np.frombuffer(buffer, dtype=dtype)
        return convert

    table = {
        "I8": np.int8,
        "I16": np.int16,
        "I32": np.int32,
        "I64": np.int64,
        "U8": np.uint8,
        "U16": np.uint16,
        "U32": np.uint32,
        "U64": np.uint64,
        "F32": np.float32,
        "F64": np.float64,
        "CF32": np.complex64,
        "CF64": np.complex128,
        "VU64": _array(np.uint64),
        "VF32": _array(np.float32),
        "VF64": _array(np.float64),
        "VCF32": _array(np.complex64),
        "VCF64": _array(np.complex128),
    }
    _jetstream_value_converter_cache.append(table)
    return table


_JETSTREAM_DTYPE_CODES = {
    ("i", 1): 10,
    ("i", 2): 11,
    ("i", 4): 12,
    ("i", 8): 13,
    ("u", 1): 14,
    ("u", 2): 15,
    ("u", 4): 16,
    ("u", 8): 17,
    ("f", 4): 18,
    ("f", 8): 19,
    ("c", 8): 20,
    ("c", 16): 21,
}


def _jetstream_classify_attribute(value):
    if value is None:
        return 7
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return 1
    if isinstance(value, float):
        return 2
    if isinstance(value, complex):
        return 6
    if isinstance(value, str):
        return 3
    if isinstance(value, dict):
        return 4
    if isinstance(value, (list, tuple)):
        return 5
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        ndim = getattr(value, "ndim", None)
        kind = getattr(dtype, "kind", "")
        code = _JETSTREAM_DTYPE_CODES.get((kind, getattr(dtype, "itemsize", 0)))
        if ndim == 1:
            if code in (17, 18, 19, 20, 21) and getattr(dtype, "isnative", False):
                flags = getattr(value, "flags", None)
                if flags is not None and getattr(flags, "c_contiguous", False):
                    return 100 + code
            return 5
        if ndim == 0:
            if kind == "b":
                return 0
            if code is not None:
                return code
    return -1
