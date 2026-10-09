#ifndef JETSTREAM_DOMAINS_CORE_COMPARATOR_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_COMPARATOR_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Comparator : public Block::Config {
    U64 inputCount = 2;
    F32 tolerance = 1e-6f;

    JST_BLOCK_TYPE(comparator);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_NODE_SIZE(M);
    JST_BLOCK_PARAMS(inputCount, tolerance);

    JST_BLOCK_DESCRIPTION(
        "Comparator",
        "Compares inputs for numerical similarity.",
        "# Comparator\n"
        "\n"
        "Checks whether one or more tensors match a reference element by element within"
        " Tolerance, and outputs their differences. It usually sits at the end of two "
        "chains that should agree, such as an old and a new version of a processing "
        "path.\n"
        "\n"
        "- **Tolerance is absolute.** It does not scale with the signal level, so loud "
        "signals need a larger value.\n"
        "- **NaN or infinity always fails.** Any difference that is not finite makes "
        "Max Diff infinite and Match read FAIL.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Reference** (in) | F32, CF32, F64, or CF64 | Any shape with at least one "
        "axis. Strided input is accepted. |\n"
        "| **Input 1** (in) | Same as Reference | Same shape and device as Reference. "
        "Input Count adds Input 2 and up. |\n"
        "| **Error** (out) | F32, or F64 for F64 and CF64 | Same shape as Reference, "
        "with its axes and attributes. Each value is the largest absolute difference "
        "across the inputs. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Input Count** | 2 | 2 to 16 | Number of ports, counting Reference. |\n"
        "| **Tolerance** | 1e-6 | 0 or more | Largest difference that still reads PASS."
        " |\n"
        "\n"
        "## Readouts\n"
        "\n"
        "| Readout | Shows |\n"
        "|---|---|\n"
        "| **Max Diff** | Largest value of Error in the latest buffer, or inf when a "
        "difference is not finite. |\n"
        "| **Mean Diff** | Mean of Error over the latest buffer. |\n"
        "| **MSE** | Mean of the squared Error values over the latest buffer. |\n"
        "| **Match** | Reads PASS when Max Diff is at or below Tolerance, and FAIL "
        "otherwise. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Identical paths | Defaults | Two copies of 8,192 complex samples | 8,192 "
        "zeros, Match PASS |\n"
        "| Rounding differences | Defaults | Two inputs that differ by up to 1e-7 | Max"
        " Diff about 1e-7, Match PASS |\n"
        "| Three versions at once | Input Count 3 | Reference and two inputs of 8,192 "
        "samples | 8,192 values, each the larger of the two differences |\n"
        "| Loud signals | Tolerance 0.001 | Two inputs near 1,000 that differ by 1e-4 |"
        " Max Diff about 1e-4, Match PASS |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that an input shape does not match the reference shape | "
        "One chain produces a different buffer size | Make every chain produce the "
        "Reference shape. |\n"
        "| The block reports that an input dtype does not match the reference, or an "
        "unsupported data type | The inputs differ in type, or one is an integer type |"
        " Convert every input to F32 or CF32 with a Cast block. |\n"
        "| The block reports that an input device does not match | One chain runs on "
        "CUDA while the block runs on the CPU | Copy that result to the CPU with a "
        "Duplicate block. |\n"
        "| The block reports that a port must be disconnected before reducing input "
        "count | A port above the new Input Count is still connected | Disconnect it, "
        "then lower Input Count. |\n"
        "| Match reads FAIL for signals that look the same | The level is large, so "
        "rounding exceeds the absolute Tolerance | Raise Tolerance in proportion to the"
        " signal level. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block takes the absolute difference between each input and the reference, "
        "keeps the largest one per element, and summarizes the result in the readouts. "
        "The model below covers one buffer.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def comparator(reference, *inputs, tolerance=1e-6):\n"
        "    error = np.max([np.abs(reference - x) for x in inputs], axis=0)\n"
        "    max_diff = error.max() if np.isfinite(error).all() else np.inf\n"
        "    match = bool(max_diff <= tolerance)\n"
        "    return error, max_diff, error.mean(), np.mean(error ** 2), match\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Duplicate** copies a GPU result to the CPU for comparison.\n"
        "- **Cast** converts integer inputs to F32 or CF32."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_COMPARATOR_BLOCK_HH
