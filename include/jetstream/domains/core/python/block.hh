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
        "Runs Python code over CPU tensors.",
        "# Python\n"
        "The Python block executes user-provided Python code on every compute "
        "cycle. The code must define a callable `compute(ctx)` function. The "
        "`ctx` object contains stable `inputs` and `outputs` mappings of NumPy "
        "arrays keyed by integer indexes (`0`, `1`, ...). Write output arrays "
        "in place.\n\n"

        "## Arguments\n"
        "- **Code**: Python source defining `compute(ctx)`.\n"
        "- **Input Count**: Number of input tensors exposed as `input0`, "
        "`input1`, ... ports and `ctx.inputs[0]`, ... arrays.\n"
        "- **Output Count**: Number of writable output tensors exposed as `output0`, "
        "`output1`, ... ports and `ctx.outputs[0]`, ... arrays.\n\n"
        "- **Output Tensor Specs**: Per-port output tensor entries with "
        "`shape`, `dtype`, and `device` fields, like `[1024]`, `F32`, `cpu`. "
        "The Python runtime currently supports CPU tensors.\n\n"

        "## Notes\n"
        "- Create this block with the Python runtime.\n"
        "- Blocks with zero inputs and/or zero outputs are valid.\n"
        "- Output ports have explicit dtype, shape, and device specs.\n"
        "- CPU tensors are supported by the Python runtime today.\n\n"

        "## Example\n"
        "```python\n"
        "def compute(ctx):\n"
        "    ctx.outputs[0][...] = ctx.inputs[0] * 2.0\n"
        "```"
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_PYTHON_BLOCK_HH
