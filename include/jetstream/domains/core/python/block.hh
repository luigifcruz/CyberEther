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
        "compute cycle with zero-copy access to the block's tensors.\n\n"

        "## Arguments\n"
        "- **Code**: Python source defining `compute(ctx)`.\n"
        "- **Input Count**: Number of input ports.\n"
        "- **Output Count**: Number of output ports.\n"
        "- **Output Tensor Specs**: Shape, data type, and device of each output "
        "tensor.\n\n"

        "## Useful For\n"
        "- Prototyping processing steps before writing a native block.\n"
        "- Math or glue logic between blocks without recompiling.\n"
        "- Using the NumPy, CuPy, and SciPy ecosystem inside a flowgraph.\n\n"

        "## Usage\n"

        "### Tensor I/O\n"
        "The `input0` and `output0` ports are exposed as `ctx.inputs[0]` and "
        "`ctx.outputs[0]`, arriving as NumPy (CPU) or CuPy (CUDA) arrays. "
        "Write outputs in place and treat inputs as read-only:\n\n"

        "```python\n"
        "def compute(ctx):\n"
        "    ctx.outputs[0][...] = ctx.inputs[0] * 2.0\n"
        "```\n\n"

        "### Tensor Attributes\n"
        "Tensor attributes like `sampleRate` travel with the data. Read them "
        "from `ctx.input_attrs[0]` and write them to `ctx.output_attrs[0]` so "
        "downstream blocks stay informed:\n\n"

        "```python\n"
        "def compute(ctx):\n"
        "    ctx.outputs[0][...] = ctx.inputs[0][::2]\n"
        "    ctx.output_attrs[0][\"sampleRate\"] = ctx.input_attrs[0][\"sampleRate\"] / 2.0\n"
        "```\n\n"

        "### Flowgraph Environment\n"
        "The flowgraph environment is shared with every block and the "
        "Environment window. Read it through `ctx.env` and publish by "
        "assigning dicts to top-level keys:\n\n"

        "```python\n"
        "def compute(ctx):\n"
        "    gain = ctx.env.get(\"station\", {}).get(\"gain\", 1.0)\n"
        "    ctx.outputs[0][...] = ctx.inputs[0] * gain\n"
        "    ctx.env[\"status\"] = {\"peak\": float(abs(ctx.outputs[0]).max())}\n"
        "```\n\n"

        "### Block Metrics\n"
        "Metrics published by other blocks are readable through "
        "`ctx.metrics`. The first read of a block's name subscribes to it "
        "and returns an empty mapping, with live values available from the "
        "next cycle on:\n\n"

        "```python\n"
        "def compute(ctx):\n"
        "    progress = ctx.metrics[\"file_reader\"].get(\"progress\")\n"
        "    ctx.metrics.subscribe_all()  # subscribe to every block\n"
        "```\n\n"

        "### Lifecycle And Errors\n"
        "Attribute and environment writes are published when `compute` "
        "returns. The code loads once at creation, globals persist across "
        "cycles, and an optional `cleanup()` runs at destroy. Output from "
        "`print()` and exceptions go to the block console, and a failing "
        "cycle is skipped without stopping the flowgraph. CUDA tensors "
        "require CuPy, and streams must be synchronized before returning."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_PYTHON_BLOCK_HH
