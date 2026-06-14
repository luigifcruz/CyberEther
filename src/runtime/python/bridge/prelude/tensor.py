import importlib as _jetstream_importlib


class _JetstreamBridge:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __getitem__(self, key):
        if key == "inputs":
            return self.inputs
        if key == "outputs":
            return self.outputs
        raise KeyError(key)


def _jetstream_tensor_from_memory(memory, dtype, shape):
    _jetstream_np = _jetstream_importlib.import_module("numpy")

    return _jetstream_np.frombuffer(memory, dtype=_jetstream_np.dtype(dtype)).reshape(shape)


def _jetstream_tensors_from_specs(
    specs,
    _tensor_from_memory=_jetstream_tensor_from_memory,
):
    return {
        index: _tensor_from_memory(memory, dtype, shape)
        for index, (memory, dtype, shape) in enumerate(specs)
    }


def _jetstream_create_bridge(
    inputs,
    outputs,
    _tensors_from_specs=_jetstream_tensors_from_specs,
):
    return _JetstreamBridge(
        _tensors_from_specs(inputs),
        _tensors_from_specs(outputs),
    )
