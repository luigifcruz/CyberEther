import importlib as _jetstream_importlib


class _JetstreamBridge:
    def __init__(self, inputs, outputs, input_attrs, output_attrs, env):
        _jetstream_types = _jetstream_importlib.import_module("types")

        self.inputs = inputs
        self.outputs = outputs
        self.input_attrs = {
            index: _jetstream_types.MappingProxyType(attrs)
            for index, attrs in enumerate(input_attrs)
        }
        self.output_attrs = dict(enumerate(output_attrs))
        self.env = env

    def __getitem__(self, key):
        if key == "inputs":
            return self.inputs
        if key == "outputs":
            return self.outputs
        if key == "input_attrs":
            return self.input_attrs
        if key == "output_attrs":
            return self.output_attrs
        if key == "env":
            return self.env
        raise KeyError(key)


def _jetstream_create_bridge(
    inputs,
    outputs,
    input_attrs,
    output_attrs,
    env,
    _tensors_from_specs=_jetstream_tensors_from_specs,
):
    return _JetstreamBridge(
        _tensors_from_specs(inputs),
        _tensors_from_specs(outputs),
        input_attrs,
        output_attrs,
        env,
    )
