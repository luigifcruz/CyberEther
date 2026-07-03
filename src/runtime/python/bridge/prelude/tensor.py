import importlib as _jetstream_importlib


_jetstream_readonly_cupy_array_types = {}


def _jetstream_readonly_cuda_array_type(_jetstream_cp):
    _jetstream_key = id(_jetstream_cp)
    _jetstream_cached = _jetstream_readonly_cupy_array_types.get(_jetstream_key)
    if _jetstream_cached is not None:
        return _jetstream_cached

    def _jetstream_readonly_error(self, *args, **kwargs):
        raise ValueError("assignment destination is read-only")

    class _JetstreamReadOnlyCuPyArray(_jetstream_cp.ndarray):
        __array_priority__ = 1000

        __setitem__ = _jetstream_readonly_error
        __iadd__ = _jetstream_readonly_error
        __isub__ = _jetstream_readonly_error
        __imul__ = _jetstream_readonly_error
        __imatmul__ = _jetstream_readonly_error
        __itruediv__ = _jetstream_readonly_error
        __ifloordiv__ = _jetstream_readonly_error
        __imod__ = _jetstream_readonly_error
        __ipow__ = _jetstream_readonly_error
        __ilshift__ = _jetstream_readonly_error
        __irshift__ = _jetstream_readonly_error
        __iand__ = _jetstream_readonly_error
        __ixor__ = _jetstream_readonly_error
        __ior__ = _jetstream_readonly_error

        fill = _jetstream_readonly_error
        put = _jetstream_readonly_error
        sort = _jetstream_readonly_error
        partition = _jetstream_readonly_error

        @property
        def __cuda_array_interface__(self):
            _jetstream_array = _jetstream_cp.ndarray.view(
                self,
                type=_jetstream_cp.ndarray,
            )
            _jetstream_interface = dict(_jetstream_array.__cuda_array_interface__)
            _jetstream_data = _jetstream_interface.get("data")
            if _jetstream_data is not None:
                _jetstream_interface["data"] = (_jetstream_data[0], True)
            return _jetstream_interface

        def byteswap(self, inplace=False):
            if inplace:
                raise ValueError("assignment destination is read-only")
            return super().byteswap(False)

        def copy(self, order="C"):
            return _jetstream_cp.ndarray.view(
                self,
                type=_jetstream_cp.ndarray,
            ).copy(order=order)

        def view(self, dtype=None, type=None):
            if dtype is _jetstream_cp.ndarray:
                dtype = None
                type = _JetstreamReadOnlyCuPyArray
            elif type is _jetstream_cp.ndarray:
                type = _JetstreamReadOnlyCuPyArray
            return super().view(dtype=dtype, type=type)

    _jetstream_readonly_cupy_array_types[_jetstream_key] = _JetstreamReadOnlyCuPyArray
    return _JetstreamReadOnlyCuPyArray


def _jetstream_tensor_from_memory(device, memory, dtype, shape, strides, writable):
    if device == "cuda":
        _jetstream_cp = _jetstream_importlib.import_module("cupy")

        pointer, span = memory
        base = _jetstream_cp.cuda.UnownedMemory(pointer, span, None)
        memptr = _jetstream_cp.cuda.MemoryPointer(base, 0)
        array = _jetstream_cp.ndarray(
            shape,
            dtype=_jetstream_cp.dtype(dtype),
            memptr=memptr,
            strides=strides,
        )
        if not writable:
            return array.view(type=_jetstream_readonly_cuda_array_type(_jetstream_cp))
        return array

    _jetstream_np = _jetstream_importlib.import_module("numpy")

    return _jetstream_np.ndarray(
        shape,
        dtype=_jetstream_np.dtype(dtype),
        buffer=memory,
        strides=strides,
    )


def _jetstream_tensors_from_specs(
    specs,
    _tensor_from_memory=_jetstream_tensor_from_memory,
):
    return {
        index: _tensor_from_memory(device, memory, dtype, shape, strides, writable)
        for index, (device, memory, dtype, shape, strides, writable) in enumerate(specs)
    }
