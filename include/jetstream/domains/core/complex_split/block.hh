#ifndef JETSTREAM_DOMAINS_CORE_COMPLEX_SPLIT_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_COMPLEX_SPLIT_BLOCK_HH

#include <string>

#include "jetstream/block.hh"
#include "jetstream/types.hh"

namespace Jetstream::Blocks {

struct ComplexSplit : public Block::Config {
    std::string mode = "rectangular";

    JST_BLOCK_TYPE(complex_split);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(mode);
    JST_BLOCK_DESCRIPTION(
        "Complex Split",
        "Splits a complex signal in two.",
        "# Complex Split\n"
        "\n"
        "Splits each complex sample into two real values, either its real and imaginary"
        " parts or its magnitude and phase. It usually follows a radio or a Filter, and"
        " feeds plots or blocks that take only F32.\n"
        "\n"
        "- **The output labels follow Mode.** Real and Imag become Magnitude and Phase.\n"
        "- **Each lane is independent.** Every batch and channel is processed "
        "separately.\n"
        "- **Phase is in radians.** It runs from -pi to pi and wraps around once per "
        "turn.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | CF32 | Any shape with at least one axis. Strided input is "
        "accepted. |\n"
        "| **Real** (out) | F32 | Same shape, with every axis and attribute kept. "
        "Labeled Magnitude in Magnitude and Phase. |\n"
        "| **Imag** (out) | F32 | Same shape, with every axis and attribute kept. "
        "Labeled Phase in Magnitude and Phase. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Mode** | Real and Imag | Real and Imag, Magnitude and Phase | Real and "
        "Imag outputs the two parts of each sample. Magnitude and Phase outputs the "
        "linear magnitude and the angle. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Plot the two parts | Defaults | 8,192 complex samples | 8,192 real parts and"
        " 8,192 imaginary parts |\n"
        "| Linear envelope | Magnitude and Phase | 8,192 complex samples | 8,192 "
        "magnitudes and 8,192 phases |\n"
        "| Batched capture | Defaults | 8 batches x 1,024 complex samples | Two outputs"
        " of 8 batches x 1,024 real samples |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports an unsupported input data type | The input is not CF32, "
        "such as CI8 samples from a file or a real signal | Convert complex integers "
        "with a Cast block. A real signal needs no split. |\n"
        "| The phase output jumps between -pi and pi | The angle wraps around once per "
        "turn | Unwrap it in a Python block, or use the FM Demodulator for frequency. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block reads each complex sample once and writes the two values to separate"
        " outputs. Magnitude and Phase uses the hypotenuse and the two-argument "
        "arctangent. The model below covers one buffer.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def complex_split(x, mode=\"Real and Imag\"):\n"
        "    if mode == \"Magnitude and Phase\":\n"
        "        return np.abs(x), np.angle(x)\n"
        "    return x.real, x.imag\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Complex Merge** builds a complex signal from two real ones.\n"
        "- **Amplitude** gives the magnitude in decibels.\n"
        "- **Cast** converts complex integers to CF32 first."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_COMPLEX_SPLIT_BLOCK_HH
