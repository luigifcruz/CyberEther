#ifndef JETSTREAM_DOMAINS_CORE_PYTHON_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_PYTHON_BLOCK_HH

#include <string>
#include <vector>

#include "jetstream/block.hh"
#include "jetstream/domains/core/python/module.hh"

namespace Jetstream::Blocks {

struct Python : public Block::Config {
    std::string code = R"PY(def compute(ctx):
    ctx.outputs[0][...] = ctx.inputs[0]
    )PY";
    U64 inputCount = 1;
    U64 outputCount = 1;
    std::vector<Modules::Python::TensorSpec> outputTensorSpecs;

    JST_BLOCK_TYPE(python);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_NODE_SIZE(XL);
    JST_BLOCK_PARAMS(code, inputCount, outputCount,
                     outputTensorSpecs);
    JST_BLOCK_DESCRIPTION(
        "Python",
        "Runs custom Python compute code.",
        "# Python\n"
        "The Python block runs a user-defined `compute(ctx)` function on every "
        "compute cycle with zero-copy access to the block's tensors. CPU tensors "
        "arrive as NumPy arrays and CUDA tensors as CuPy arrays. Globals persist "
        "across cycles, so accumulators and precomputed tables can live at module "
        "level.\n\n"

        "## Arguments\n"
        "- **Code**: Python source defining `compute(ctx)`. `ctx.inputs[0]` and "
        "`ctx.outputs[0]` correspond to the `input0` and `output0` ports. Write "
        "outputs in place (`ctx.outputs[0][...] = value`); rebinding the name "
        "writes nothing.\n"
        "- **Input Count**: Number of input ports.\n"
        "- **Output Count**: Number of output ports.\n"
        "- **Output Tensor Specs**: Shape, data type, and device of each output "
        "tensor. Code cannot change them at runtime.\n\n"

        "## Useful For\n"
        "- Prototyping processing steps before writing a native block.\n"
        "- Math or glue logic between blocks without recompiling.\n"
        "- Using the NumPy, CuPy, and SciPy ecosystem inside a flowgraph.\n\n"

        "## Examples\n"
        "```python\n"
        "def compute(ctx):\n"
        "    ctx.outputs[0][...] = ctx.inputs[0] * 2.0\n"
        "```\n\n"

        "## Implementation\n"
        "1. The code is loaded once at block creation; imports resolve against "
        "the selected Python runtime's environment.\n"
        "2. Inputs are read-only views; copy before mutating.\n"
        "3. `print()` output and exceptions appear in the block console; a "
        "failing `compute` skips the cycle without stopping the flowgraph.\n"
        "4. An optional `cleanup()` function runs when the block is destroyed.\n"
        "5. CUDA tensors require CuPy. Inputs are complete when `compute` "
        "starts, but CuPy launches are asynchronous: synchronize before "
        "returning (e.g. `cupy.cuda.Stream.null.synchronize()`) so downstream "
        "blocks see finished writes."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_PYTHON_BLOCK_HH
